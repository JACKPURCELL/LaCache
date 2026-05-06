"""
exp3_assumption_ii.py — RQ1, Assumption (ii): Semantic Separation
=================================================================
Assumption (ii) (Semantic Separation): for an adversarial query x* whose
ISR judge marks SUCCESS and the victim query x, their LLM responses are
separated in embedding space by more than delta_2 / C:

    d(f(x*), f(x)) > delta_2 / C,  with C in (0, 1)

For each (victim_query, source_prompt) pair from the dataset:
  1. Generate the full-length LLM response f(x)  to the victim query
  2. Generate the full-length LLM response f(x*) to the adversarial query
  3. Measure semantic distance d = 1 - cos_sim(emb(f(x)), emb(f(x*)))

Compares against delta_2 measured by exp2_lsh_calibration.py.

Outputs (under outputs_<track>/exp3/<llm>/):
  summary.json   mean / P5 / hit-rate of d vs delta_2
  raw_data.json  per-record (q, q_star, response_q, response_qstar, d)
  histogram.png  d distribution with delta_2 marker (Figure 3(b))
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from common import (
    load_sc_ipi, load_sc_marco, vllm_generate, vllm_chat_format, embed_texts,
    cosine_sim, DEFAULT_EMBEDDING_MODEL, DEFAULT_BACKEND_LLM, OUTPUTS_DIR,
    save_json, save_csv,
)


def run_exp3(max_records=None, max_response_tokens=200,
             embedding_model=DEFAULT_EMBEDDING_MODEL,
             backend_llm=DEFAULT_BACKEND_LLM,
             output_prefix="exp3",
             track="kca",
             ndss_method="prompt_injection_white_box",
             dataset_path=None,
             require_isr_success=False):

    print(f"\n=== Experiment 3: Assumption (ii) verification ({track}) ===")
    print(f"Embedding: {embedding_model}")
    print(f"Backend LLM: {backend_llm}")
    if track == "ndss":
        print(f"NDSS method: {ndss_method}")
    if require_isr_success:
        print(f"ISR filter: only records with isr_verdict == "
              f"{'SUCCESS' if track == 'ndss' else 'DIFFERENT'}")
    print()

    if track == "ndss":
        records = load_sc_marco(path=dataset_path, limit=max_records)
        records = [r for r in records if r.get("ndss_attacks", {}).get(ndss_method)]
        if require_isr_success:
            n_before = len(records)
            records = [r for r in records
                       if (r.get("isr_results", {}).get(ndss_method) or {})
                            .get("isr_verdict") == "SUCCESS"]
            print(f"  ISR filter: kept {len(records)}/{n_before}")
    else:
        records = load_sc_ipi(path=dataset_path, limit=max_records)
        records = [r for r in records if r.get("source_prompt")]
        if require_isr_success:
            n_before = len(records)
            records = [r for r in records
                       if r.get("isr_verdict") == "DIFFERENT"]
            print(f"  ISR filter: kept {len(records)}/{n_before}")
    print(f"Records: {len(records)}")

    # Batch generate responses
    victim_prompts = [vllm_chat_format(r["victim_query"]) for r in records]
    if track == "ndss":
        source_prompts = [vllm_chat_format(r["ndss_attacks"][ndss_method])
                          for r in records]
    else:
        source_prompts = [vllm_chat_format(r["source_prompt"]) for r in records]

    print("\n[gen] generating victim responses...")
    victim_responses = vllm_generate(
        victim_prompts, max_tokens=max_response_tokens, temperature=0.0,
        model_name=backend_llm,
    )

    print("[gen] generating source responses...")
    source_responses = vllm_generate(
        source_prompts, max_tokens=max_response_tokens, temperature=0.0,
        model_name=backend_llm,
    )

    # Embed
    print("[emb] embedding victim responses...")
    victim_embs = embed_texts(victim_responses, model_name=embedding_model,
                              show_progress=True)
    print("[emb] embedding source responses...")
    source_embs = embed_texts(source_responses, model_name=embedding_model,
                              show_progress=True)

    # Compute distances
    distances = 1.0 - cosine_sim(victim_embs, source_embs)

    # Per-record output
    details = []
    for i, r in enumerate(records):
        adv_text = (r["ndss_attacks"][ndss_method] if track == "ndss"
                    else r["source_prompt"])
        details.append({
            "id": r["id"],
            "category": r.get("category", "?"),
            "victim_query": r["victim_query"],
            "source_prompt": adv_text,
            "victim_response": victim_responses[i],
            "source_response": source_responses[i],
            "response_distance": float(distances[i]),
        })

    # Stats
    dist_arr = np.asarray([d["response_distance"] for d in details])
    stats_overall = {
        "n": int(len(dist_arr)),
        "mean": float(dist_arr.mean()),
        "std": float(dist_arr.std()),
        "p05": float(np.percentile(dist_arr, 5)),
        "p50": float(np.percentile(dist_arr, 50)),
        "p95": float(np.percentile(dist_arr, 95)),
        "min": float(dist_arr.min()),
        "max": float(dist_arr.max()),
    }

    # Per-category
    by_cat = {}
    for d in details:
        c = d["category"]
        by_cat.setdefault(c, []).append(d["response_distance"])
    stats_by_cat = {
        c: {
            "n": len(v),
            "mean": float(np.mean(v)),
            "p05": float(np.percentile(v, 5)),
            "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)),
        }
        for c, v in by_cat.items()
    }

    # Compare to delta_2 from exp2 if available
    delta_2 = None
    exp2_summary_path = OUTPUTS_DIR / "exp2_summary.json"
    if exp2_summary_path.exists():
        exp2 = json.load(open(exp2_summary_path))
        delta_2 = exp2["calibrated_lsh_params"]["delta_2"]

    # Print
    print("\n" + "=" * 60)
    print("Assumption (ii) verification — Response distance statistics")
    print("=" * 60)
    print(f"n={stats_overall['n']}")
    print(f"mean = {stats_overall['mean']:.4f}")
    print(f"p05  = {stats_overall['p05']:.4f}")
    print(f"p50  = {stats_overall['p50']:.4f}")
    print(f"p95  = {stats_overall['p95']:.4f}")

    if delta_2 is not None:
        print(f"\nCalibrated delta_2 from exp2: {delta_2:.4f}")
        frac_above = float((dist_arr > delta_2).mean())
        print(f"Fraction above delta_2: {frac_above:.1%}")
        print(f"Assumption (ii) holds for {frac_above:.1%} of records.")

    print("\nBy category:")
    print(f"{'category':<25s} {'n':>5s} {'mean':>8s} {'p05':>8s}")
    for c, s in sorted(stats_by_cat.items()):
        print(f"{c:<25s} {s['n']:>5d} {s['mean']:>8.4f} {s['p05']:>8.4f}")

    # Save
    summary = {
        "stats": stats_overall,
        "by_category": stats_by_cat,
        "delta_2_from_exp2": delta_2,
        "fraction_above_delta_2": (
            float((dist_arr > delta_2).mean()) if delta_2 is not None else None
        ),
    }
    save_json(summary, OUTPUTS_DIR / f"{output_prefix}_summary.json")
    save_json(details, OUTPUTS_DIR / f"{output_prefix}_details.json")
    save_csv(pd.DataFrame([
        {"id": d["id"], "category": d["category"], "response_distance": d["response_distance"]}
        for d in details
    ]), OUTPUTS_DIR / f"{output_prefix}_details.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(dist_arr, bins=40, alpha=0.7, color="purple")
    ax.axvline(stats_overall["p05"], color="black", linestyle=":",
               label=f"p05 = {stats_overall['p05']:.3f}")
    ax.axvline(stats_overall["p50"], color="black", linestyle="--",
               label=f"p50 = {stats_overall['p50']:.3f}")
    if delta_2 is not None:
        ax.axvline(delta_2, color="red", linestyle="-",
                   label=f"delta_2 (from exp2) = {delta_2:.3f}")
    ax.set_xlabel("Response semantic distance Delta(d(x*), d(x))")
    ax.set_ylabel("Count")
    ax.set_title("Assumption (ii): response separation for adversarial pairs")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"{output_prefix}_histogram.png", dpi=150)
    print(f"[saved] {output_prefix}_histogram.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--embedding", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--backend", type=str, default=DEFAULT_BACKEND_LLM)
    parser.add_argument("--track", type=str, default="kca", choices=["kca", "ndss"])
    parser.add_argument("--ndss-method", type=str,
                        default="prompt_injection_white_box",
                        choices=["zero_shot", "in_context_learning", "prompt_injection",
                                 "zero_shot_white_box", "in_context_learning_white_box",
                                 "prompt_injection_white_box"])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default="exp3")
    parser.add_argument("--require-isr-success", action="store_true",
                        help="Filter records to ISR=SUCCESS (NDSS) / "
                             "verdict=DIFFERENT (KCA) before generating.")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.max_records = 30

    run_exp3(
        max_records=args.max_records,
        max_response_tokens=args.max_tokens,
        embedding_model=args.embedding,
        backend_llm=args.backend,
        output_prefix=args.output_prefix,
        track=args.track,
        ndss_method=args.ndss_method,
        dataset_path=args.dataset_path,
        require_isr_success=args.require_isr_success,
    )


if __name__ == "__main__":
    main()
