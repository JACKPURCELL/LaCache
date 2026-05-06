"""
exp5_lacache_defense.py — RQ2: LaCache Effectiveness vs. Baseline Defenses
==========================================================================
Main defense experiment. Reports Attack Hit Rate (AHR) and Benign Hit Rate
(BHR) for LaCache and a query-side baseline defense across the two tracks.

Defenses evaluated:
  D1. no_defense           cache check  sim(x, x') >= tau_c  only (undefended)
  D2. lacache (ours)       cache check AND lookahead check  sim(f_k(x), f_k(x')) >= tau_a
  D3. paraphrase_recheck   LLM paraphrases the lookup query and re-checks cache match

Workloads:
  - Attack pair:  cached = adversarial_query x*,  lookup = victim_query x
  - Benign pair:  cached = paraphrase q1,         lookup = paraphrase q2

AHR is reported on the ISR=SUCCESS subset (i.e. only attacks that the LLM
actually obeys); BHR is reported on the full benign paraphrase pool.

Outputs (under outputs_<track>/exp5/<llm>/):
  summary.json   AHR / BHR per defense (consumed by paper Table 1, Table 2)
  raw_data.json  per-record similarities and per-defense decisions
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from common import (
    DEFAULT_K, GUARD_THRESHOLD_FIXED, add_common_args, cosine_sim,
    embed_texts, get_cache_threshold, gpt_call_parallel, load_benign_pairs,
    load_sc_ipi, load_sc_marco, resolve_llm, save_csv, save_json,
    track_output_dir, vllm_chat,
)


PARAPHRASE_PROMPT = """Rewrite the following question as a natural paraphrase
that asks the same essential thing in different words. Keep the same intent.
Output ONLY the paraphrased question, no quotes or preamble.

Original: {query}

Paraphrase:"""


# =============================================================================
# Workload extraction
# =============================================================================

def collect_attack_pairs(track: str, limit: int, ndss_method: str,
                          dataset_path: str = None) -> List[Dict[str, Any]]:
    out = []
    if track == "kca":
        records = load_sc_ipi(path=dataset_path, limit=limit)
        for r in records:
            if r.get("adv_query"):
                out.append({
                    "id": r["id"],
                    "victim_query": r["victim_query"],
                    "victim_query_paraphrase": r.get("victim_query_paraphrase", ""),
                    "adv_query": r["adv_query"],
                    "category": r.get("category"),
                })
    elif track == "ndss":
        records = load_sc_marco(path=dataset_path, limit=limit)
        for r in records:
            adv = r.get("ndss_attacks", {}).get(ndss_method)
            if adv:
                out.append({
                    "id": r["id"],
                    "victim_query": r["victim_query"],
                    "victim_query_paraphrase": r.get("victim_query_paraphrase", ""),
                    "adv_query": adv,
                    "category": r.get("category"),
                    "ndss_method": ndss_method,
                })
    else:
        raise ValueError(track)
    return out


# =============================================================================
# Defenses
# =============================================================================

def d1_no_defense(sims_q: np.ndarray, tau_cache: float) -> np.ndarray:
    return sims_q >= tau_cache


def d2_lacache(sims_q: np.ndarray, sims_r: np.ndarray,
                tau_cache: float, tau_guard: float) -> np.ndarray:
    return (sims_q >= tau_cache) & (sims_r >= tau_guard)


def d3_paraphrase_recheck(victim_q: List[str], cached_q: List[str],
                           paraphrase_lookup: List[str],
                           tau_cache: float, embedding: str,
                           base_admit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Embed paraphrased lookup_q, recheck sim vs cached_q."""
    e_p = embed_texts(paraphrase_lookup, model_name=embedding)
    e_c = embed_texts(cached_q, model_name=embedding)
    sims = np.asarray(cosine_sim(e_p, e_c))
    return base_admit & (sims >= tau_cache), sims


# =============================================================================
# Driver
# =============================================================================

def run_exp5(*, track: str, llm: str, embedding: str, k: int,
             limit: int, num_benign: int, ndss_method: str,
             paraphrase_workers: int, output_prefix: str,
             dataset_path: str = None,
             run_paraphrase_recheck: bool = False):
    out_dir = track_output_dir(track) / output_prefix / llm
    out_dir.mkdir(parents=True, exist_ok=True)
    full_llm = resolve_llm(llm)

    tau_cache = get_cache_threshold(embedding, track)
    tau_guard = GUARD_THRESHOLD_FIXED
    print(f"=== Experiment 5: LaCache + paraphrase-recheck baseline ===")
    print(f"track={track}  llm={full_llm}  emb={embedding}  k={k}")
    print(f"tau_c={tau_cache:.3f}  tau_a={tau_guard:.3f}")

    # ---- attack workload ----
    attacks = collect_attack_pairs(track, limit, ndss_method, dataset_path=dataset_path)
    n_atk = len(attacks)
    if n_atk == 0:
        raise SystemExit("no attack pairs")

    victim_q_atk = [a["victim_query"] for a in attacks]
    cached_q_atk = [a["adv_query"] for a in attacks]
    print(f"\n[attack] {n_atk} pairs")

    # Embeddings + sims (query side)
    e_v = embed_texts(victim_q_atk, model_name=embedding)
    e_c = embed_texts(cached_q_atk, model_name=embedding)
    q_sims_atk = np.asarray(cosine_sim(e_v, e_c))

    # Response prefixes for guard
    print(f"[attack] generating responses (k={k})")
    resp_v_atk = vllm_chat(victim_q_atk, max_tokens=k, model_name=full_llm)
    resp_c_atk = vllm_chat(cached_q_atk, max_tokens=k, model_name=full_llm)
    er_v = embed_texts(resp_v_atk, model_name=embedding)
    er_c = embed_texts(resp_c_atk, model_name=embedding)
    r_sims_atk = np.asarray(cosine_sim(er_v, er_c))

    # ---- per-record paraphrase side (same victim, benign rephrasing) ----
    para_q_atk = [(a.get("victim_query_paraphrase") or a["victim_query"])
                   for a in attacks]
    has_para = [bool(a.get("victim_query_paraphrase")) for a in attacks]
    n_with_para = sum(has_para)
    print(f"[attack] generating paraphrase responses (k={k})  "
          f"{n_with_para}/{n_atk} records have a paraphrase")
    resp_p_atk = vllm_chat(para_q_atk, max_tokens=k, model_name=full_llm)
    e_para = embed_texts(para_q_atk, model_name=embedding)
    er_p = embed_texts(resp_p_atk, model_name=embedding)
    q_sims_para = np.asarray(cosine_sim(e_v, e_para))
    r_sims_para = np.asarray(cosine_sim(er_v, er_p))

    # ---- benign workload ----
    benign_pairs = load_benign_pairs(num_benign)
    n_b = len(benign_pairs)
    cached_q_b = [p[0] for p in benign_pairs]
    lookup_q_b = [p[1] for p in benign_pairs]

    e_b1 = embed_texts(cached_q_b, model_name=embedding)
    e_b2 = embed_texts(lookup_q_b, model_name=embedding)
    q_sims_b = np.asarray(cosine_sim(e_b1, e_b2))

    print(f"[benign] generating responses (k={k})  n_benign={n_b}")
    resp_b1 = vllm_chat(cached_q_b, max_tokens=k, model_name=full_llm)
    resp_b2 = vllm_chat(lookup_q_b, max_tokens=k, model_name=full_llm)
    er_b1 = embed_texts(resp_b1, model_name=embedding)
    er_b2 = embed_texts(resp_b2, model_name=embedding)
    r_sims_b = np.asarray(cosine_sim(er_b1, er_b2))

    # ---- paraphrase-and-recheck baseline ----
    # Skipped by default — the GPT paraphrasing pass is the most expensive
    # baseline to run. Pass --paraphrase-recheck to enable.
    if run_paraphrase_recheck:
        print(f"\n[paraphrase] gpt paraphrasing {n_atk + n_b} lookup queries")
        para_lookup_atk = gpt_call_parallel(
            [PARAPHRASE_PROMPT.format(query=q) for q in victim_q_atk],
            max_workers=paraphrase_workers,
        )
        para_lookup_atk = [(p or victim_q_atk[i]).strip()
                           for i, p in enumerate(para_lookup_atk)]
        para_lookup_b = gpt_call_parallel(
            [PARAPHRASE_PROMPT.format(query=q) for q in lookup_q_b],
            max_workers=paraphrase_workers,
        )
        para_lookup_b = [(p or lookup_q_b[i]).strip()
                         for i, p in enumerate(para_lookup_b)]
    else:
        print("\n[paraphrase] skipped (pass --paraphrase-recheck to enable)")
        para_lookup_atk = victim_q_atk
        para_lookup_b = lookup_q_b

    # ---- evaluate defenses on attack workload ----
    a_d1 = d1_no_defense(q_sims_atk, tau_cache)
    a_d2 = d2_lacache(q_sims_atk, r_sims_atk, tau_cache, tau_guard)

    # ---- evaluate defenses on benign workload ----
    b_d1 = d1_no_defense(q_sims_b, tau_cache)
    b_d2 = d2_lacache(q_sims_b, r_sims_b, tau_cache, tau_guard)

    if run_paraphrase_recheck:
        a_d3, a_d3_sims = d3_paraphrase_recheck(
            victim_q_atk, cached_q_atk, para_lookup_atk, tau_cache, embedding, a_d1,
        )
        b_d3, b_d3_sims = d3_paraphrase_recheck(
            lookup_q_b, cached_q_b, para_lookup_b, tau_cache, embedding, b_d1,
        )
    else:
        a_d3 = np.zeros_like(a_d1)
        b_d3 = np.zeros_like(b_d1)
        a_d3_sims = np.zeros(len(a_d1))
        b_d3_sims = np.zeros(len(b_d1))

    defenses = {
        "no_defense": {
            "attack_HR": float(a_d1.mean()), "benign_HR": float(b_d1.mean()),
        },
        "lacache_k%d" % k: {
            "attack_HR": float(a_d2.mean()), "benign_HR": float(b_d2.mean()),
        },
    }
    if run_paraphrase_recheck:
        defenses["paraphrase_recheck"] = {
            "attack_HR": float(a_d3.mean()), "benign_HR": float(b_d3.mean()),
        }

    print("\n=== Defense table ===")
    print(f"{'defense':<22s} {'attack HR':>10s} {'benign HR':>10s}")
    for name, d in defenses.items():
        print(f"{name:<22s} {d['attack_HR']:>10.3f} {d['benign_HR']:>10.3f}")

    # Per-record paraphrase summary (same victim, benign rephrasing)
    if any(has_para):
        para_q_pass = float(((q_sims_para >= tau_cache) & np.asarray(has_para)).sum()
                              / max(1, n_with_para))
        para_lacache_admit = ((q_sims_para >= tau_cache) & (r_sims_para >= tau_guard)
                                & np.asarray(has_para))
        para_lacache_pass = float(para_lacache_admit.sum() / max(1, n_with_para))
        print(f"\n[per-record paraphrase, n={n_with_para}, same victim as attack]")
        print(f"  cache_only_pass        = {para_q_pass:.3f}  "
              f"(would benign rephrase be cached?)")
        print(f"  lacache_pass           = {para_lacache_pass:.3f}  "
              f"(would benign rephrase pass cache+guard?)")

    # By-category breakdown for KCA / SCP (LaCache only — main result)
    by_cat: Dict[str, Dict[str, int]] = {}
    for i, atk in enumerate(attacks):
        c = atk.get("category") or "?"
        if c not in by_cat:
            by_cat[c] = {"total": 0, "lacache_hits": 0, "no_defense_hits": 0}
        by_cat[c]["total"] += 1
        if a_d1[i]:
            by_cat[c]["no_defense_hits"] += 1
        if a_d2[i]:
            by_cat[c]["lacache_hits"] += 1
    print("\nLaCache attack-HR by category:")
    for c in sorted(by_cat):
        s = by_cat[c]
        print(f"  {c:<24s} lacache={s['lacache_hits']}/{s['total']}  "
              f"baseline={s['no_defense_hits']}/{s['total']}")

    summary = {
        "track": track, "llm": full_llm, "embedding": embedding, "k": k,
        "tau_cache": tau_cache, "tau_guard": tau_guard,
        "n_attacks": n_atk, "n_benign": n_b,
        "ndss_method": ndss_method if track == "ndss" else None,
        "defenses": defenses,
        "by_category": by_cat,
    }
    # Per-record human-readable details (texts + responses) so you can read
    # exactly what the attacker / paraphrase / victim saw and produced.
    attack_records = []
    for i, a in enumerate(attacks):
        attack_records.append({
            "id": a["id"],
            "category": a.get("category"),
            "victim_query": a["victim_query"],
            "victim_query_paraphrase": a.get("victim_query_paraphrase", ""),
            "adv_query": a["adv_query"],
            "ndss_method": a.get("ndss_method"),
            "llm_response_victim": resp_v_atk[i],
            "llm_response_adv": resp_c_atk[i],
            "llm_response_paraphrase": (resp_p_atk[i] if has_para[i] else None),
            "query_sim_adv": float(q_sims_atk[i]),
            "response_sim_adv": float(r_sims_atk[i]),
            "query_sim_paraphrase": (
                float(q_sims_para[i]) if has_para[i] else None),
            "response_sim_paraphrase": (
                float(r_sims_para[i]) if has_para[i] else None),
            "paraphrase_recheck_sim": (
                float(a_d3_sims[i]) if run_paraphrase_recheck else None),
            "no_defense_admit": bool(a_d1[i]),
            "lacache_admit": bool(a_d2[i]),
            "paraphrase_recheck_admit": (
                bool(a_d3[i]) if run_paraphrase_recheck else None),
            "lacache_admit_paraphrase": (
                bool(q_sims_para[i] >= tau_cache and r_sims_para[i] >= tau_guard)
                if has_para[i] else None),
        })
    benign_records = []
    for i in range(n_b):
        benign_records.append({
            "q1_cached": cached_q_b[i],
            "q2_lookup": lookup_q_b[i],
            "llm_response_q1": resp_b1[i],
            "llm_response_q2": resp_b2[i],
            "query_sim": float(q_sims_b[i]),
            "response_sim": float(r_sims_b[i]),
            "paraphrase_recheck_sim": (
                float(b_d3_sims[i]) if run_paraphrase_recheck else None),
            "no_defense_admit": bool(b_d1[i]),
            "lacache_admit": bool(b_d2[i]),
            "paraphrase_recheck_admit": (
                bool(b_d3[i]) if run_paraphrase_recheck else None),
        })
    save_json(attack_records, out_dir / "attack_details.json")
    save_json(benign_records, out_dir / "benign_details.json")

    raw = {
        "track": track,
        "embedding": embedding,
        "tau_cache": tau_cache, "tau_guard": tau_guard,
        "attack": {
            "ids": [a["id"] for a in attacks],
            "categories": [a.get("category") for a in attacks],
            "q_sims": q_sims_atk.tolist(),
            "r_sims": r_sims_atk.tolist(),
            "q_sims_paraphrase": q_sims_para.tolist(),
            "r_sims_paraphrase": r_sims_para.tolist(),
            "has_paraphrase": [bool(x) for x in has_para],
            "paraphrase_recheck_sims": a_d3_sims.tolist(),
            "victim_queries": victim_q_atk,
            "victim_query_paraphrases": para_q_atk,
            "adv_queries": cached_q_atk,
            "victim_responses": resp_v_atk,
            "paraphrase_responses": resp_p_atk,
            "adv_responses": resp_c_atk,
            "decisions": {
                "no_defense": a_d1.tolist(),
                "lacache": a_d2.tolist(),
                "paraphrase_recheck": a_d3.tolist(),
            },
        },
        "benign": {
            "q_sims": q_sims_b.tolist(),
            "r_sims": r_sims_b.tolist(),
            "paraphrase_recheck_sims": b_d3_sims.tolist(),
            "q1_cached": cached_q_b,
            "q2_lookup": lookup_q_b,
            "responses_q1": resp_b1,
            "responses_q2": resp_b2,
            "decisions": {
                "no_defense": b_d1.tolist(),
                "lacache": b_d2.tolist(),
                "paraphrase_recheck": b_d3.tolist(),
            },
        },
    }
    save_json(summary, out_dir / "summary.json")
    save_json(raw, out_dir / "raw_data.json")

    df_rows = [{"defense": name, **d} for name, d in defenses.items()]
    save_csv(pd.DataFrame(df_rows), out_dir / "defense_table.csv")


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--num-benign", type=int, default=200)
    parser.add_argument("--ndss-method", type=str, default="prompt_injection",
                        choices=["zero_shot", "in_context_learning", "prompt_injection",
                                 "zero_shot_white_box", "in_context_learning_white_box",
                                 "prompt_injection_white_box"])
    parser.add_argument("--paraphrase-workers", type=int, default=8)
    parser.add_argument("--paraphrase-recheck", dest="paraphrase_recheck",
                        action="store_true", default=False,
                        help="enable paraphrase-and-recheck baseline (off by default; expensive GPT calls)")
    parser.add_argument("--output-prefix", type=str, default="exp5")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="path to data/sc_ipi_v3.jsonl (KCA) or data/scp.jsonl (SCP); "
                             "if omitted, uses the canonical default from common.load_sc_*")
    args = parser.parse_args()

    run_exp5(
        track=args.track, llm=args.llm, embedding=args.embedding, k=args.k,
        limit=args.limit, num_benign=args.num_benign,
        ndss_method=args.ndss_method,
        paraphrase_workers=args.paraphrase_workers,
        output_prefix=args.output_prefix,
        dataset_path=args.dataset_path,
        run_paraphrase_recheck=args.paraphrase_recheck,
    )


if __name__ == "__main__":
    main()
