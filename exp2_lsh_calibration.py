"""
exp2_lsh_calibration.py — RQ1, Assumption (i): Gapped LSH (Embedding Fidelity)
==============================================================================
Empirically estimate the gap parameters (delta_1, delta_2, p_1, p_2) of the
embedding model + cache threshold tau_c configuration used by LaCache.

Assumption (i) (Embedding Fidelity / Gapped LSH):
    p_1 = P[hit | d <= delta_1]   on benign paraphrase pairs
    p_2 = P[hit | d >= delta_2]   on random / adversarial pairs
with delta_1 < delta_2 and p_1 > p_2. The script measures three distance
distributions and runs a percentile sweep to locate the operating gap.

Pair types measured:
  (A) Benign paraphrase pairs (same intent, different wording).
      Source: data/benign_pairs.jsonl. delta_1 ~= P95 of these distances.
  (B) Random pairs (two unrelated benign queries).
      Source: load_nq_queries / load_msmarco_queries. Natural baseline.
  (C) Adversarial pairs (KCA: victim_query vs. source_prompt).
      Source: data/sc_ipi_v3.jsonl. delta_2 ~= P5 of these distances.

Outputs (under outputs_<track>/exp2/):
  summary.json     stats + percentile sweep + main-theorem detection bound
  raw_data.json    per-pair distances (consumed by Figure 3(a))
  histogram.png    three distributions overlaid with tau_c marker
  gap_sweep.png    delta_1, delta_2 vs tail mass alpha
  tails.png        log-scale tail probabilities

Usage:
  python exp2_lsh_calibration.py --track kca --embedding BAAI/bge-large-en-v1.5
  python exp2_lsh_calibration.py --track kca --num-pairs 500   # default 500
  python exp2_lsh_calibration.py --quick                       # 50/50/50 smoke
"""

from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

from common import (
    DEFAULT_EMBEDDING_MODEL, add_common_args, cosine_sim, embed_texts,
    get_cache_threshold, load_benign_pairs, load_nq_queries, load_sc_ipi,
    load_sc_marco, save_csv, save_json, track_output_dir,
)


def _pair_distances(pairs: List[Tuple[str, str]],
                    embedding_model: str) -> np.ndarray:
    """For each (a, b) pair, return cosine distance = 1 - cos_sim."""
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    e_a = embed_texts(a, model_name=embedding_model, show_progress=True)
    e_b = embed_texts(b, model_name=embedding_model, show_progress=True)
    sims = np.array([cosine_sim(e_a[i], e_b[i]) for i in range(len(pairs))])
    return 1.0 - sims


def _stats(arr: np.ndarray, name: str) -> dict:
    arr = np.asarray(arr)
    out = {
        "name": name,
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    for p in [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 98, 99]:
        out[f"p{p:02d}"] = float(np.percentile(arr, p))
    return out


def run(track: str, num_pairs: int, embedding: str, threshold: float,
        seed: int, dataset_path: str, output_prefix: str = "exp2",
        ndss_method: str = "prompt_injection_white_box"):
    out_dir = track_output_dir(track) / output_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== exp2: Assumption (i) Gapped LSH calibration ===")
    print(f"track={track}  embedding={embedding}  τ_cache={threshold:.3f}")
    print(f"n_pairs per type = {num_pairs}")

    # ---------- (A) Benign paraphrase pairs (reuse outputs/benign_pairs.jsonl) ----------
    print("\n[A] loading benign paraphrase pairs from outputs/benign_pairs.jsonl")
    benign_pairs = load_benign_pairs(num_pairs)
    print(f"  {len(benign_pairs)} pairs")
    d_benign = _pair_distances(benign_pairs, embedding)

    # ---------- (B) Random benign pairs from NQ pool ----------
    print("\n[B] sampling random benign pairs from NQ pool")
    nq_pool = load_nq_queries(num_pairs * 2 + 100, seed=seed + 7)
    rng = random.Random(seed + 17)
    rng.shuffle(nq_pool)
    random_pairs = [(nq_pool[2 * i], nq_pool[2 * i + 1]) for i in range(num_pairs)]
    d_random = _pair_distances(random_pairs, embedding)

    # ---------- (C) Adversarial pairs (DIAGNOSTIC ONLY, not part of LSH sweep) ----------
    # Note: KCA source_prompt was sampled with a sim ∈ [0.5, 0.75] filter, so
    # the resulting distance distribution has artificial cutoffs and is NOT
    # representative of "random unrelated query" distance.
    # Similarly, NDSS ndss_attacks[*] contains victim_query verbatim, not a
    # random pair.
    # The Gapped LSH property is a property of the EMBEDDING, validated by
    # comparing (A) paraphrase pairs (similar) vs (B) truly random pairs
    # (dissimilar). δ₂ is taken from (B) random p(α), NOT from (C).
    if track == "ndss":
        print(f"\n[C diagnostic] loading NDSS (victim_query, ndss_attacks[{ndss_method}]) "
              f"pairs from {dataset_path}")
        records = load_sc_marco(path=dataset_path)
        if len(records) > num_pairs:
            records = random.Random(seed + 23).sample(records, num_pairs)
        adv_pairs = [(r["victim_query"], r["ndss_attacks"][ndss_method])
                     for r in records
                     if r.get("victim_query")
                     and r.get("ndss_attacks", {}).get(ndss_method)]
    else:
        print("\n[C diagnostic] loading KCA (victim_query, source_prompt) pairs from "
              f"{dataset_path}")
        records = load_sc_ipi(path=dataset_path)
        if len(records) > num_pairs:
            records = random.Random(seed + 23).sample(records, num_pairs)
        adv_pairs = [(r["victim_query"], r["source_prompt"]) for r in records
                     if r.get("victim_query") and r.get("source_prompt")]
    print(f"  {len(adv_pairs)} pairs (diagnostic only — has sim filter / "
          f"victim substring; does NOT enter the Gapped-LSH sweep)")
    if not adv_pairs:
        print("  (no diagnostic pairs found; continuing with (A)+(B) only)")
        d_adv = np.array([])
    else:
        d_adv = _pair_distances(adv_pairs, embedding)

    # ---------- Stats ----------
    stats_benign = _stats(d_benign, "benign_paraphrase")
    stats_random = _stats(d_random, "random_benign")
    stats_adv = _stats(d_adv, "adversarial") if len(d_adv) else None

    # ---------- Percentile sweep for Gapped LSH (δ₁, δ₂, p₁, p₂) ----------
    # Gapped LSH is an EMBEDDING property:
    #   - (A) paraphrase pairs are SIMILAR  → δ₁ = upper tail of (A)
    #   - (B) random pairs are DISSIMILAR   → δ₂ = lower tail of (B)
    # We do NOT use (C) adversarial pairs here — those are filtered/contain
    # the victim and are not representative of "random unrelated query".
    tau_dist = 1.0 - threshold
    sweep = []
    for alpha in [1, 2, 5, 10, 15, 20, 25, 30, 40, 50]:
        d1 = float(np.percentile(d_benign, 100 - alpha))
        d2 = float(np.percentile(d_random, alpha))
        # p₁ = P(hash collide | d ≤ δ₁) measured on (A) paraphrase
        m_b = (d_benign <= d1)
        p1 = float((d_benign[m_b] <= tau_dist).mean()) if m_b.any() else None
        # p₂ = P(hash collide | d ≥ δ₂) measured on (B) random
        m_r = (d_random >= d2)
        p2 = float((d_random[m_r] <= tau_dist).mean()) if m_r.any() else None
        sweep.append({
            "alpha_pct": alpha,
            "coverage_pct": 100 - alpha,
            "delta_1": d1,
            "delta_2": d2,
            "gap": d2 - d1,
            "gap_exists": d1 < d2,
            "p_1": p1,
            "p_2": p2,
            "p1_minus_p2": (p1 - p2) if (p1 is not None and p2 is not None) else None,
            "assumption_i_holds": (p1 is not None and p2 is not None
                                    and p1 > p2 and d1 < d2),
            "detection_bound_1_minus_p2": (1.0 - p2) if p2 is not None else None,
        })

    # Find smallest α (most permissive coverage) where gap exists
    critical_alpha = next((r["alpha_pct"] for r in sweep if r["gap_exists"]), None)

    # Headline operating point: α=5 (paper-standard "robust 95% coverage on both sides")
    headline = next((r for r in sweep if r["alpha_pct"] == 5), sweep[0])

    summary = {
        "track": track,
        "embedding": embedding,
        "threshold_sim": threshold,
        "threshold_dist_tau": tau_dist,
        "n_per_type": num_pairs,
        "stats": {
            "benign_paraphrase": stats_benign,
            "random_benign": stats_random,
            "adversarial": stats_adv,
        },
        "sweep": sweep,
        "critical_alpha": critical_alpha,
        "headline_alpha_5": headline,
        "main_theorem_detection_bound": (
            (1.0 - headline["p_2"]) if headline["p_2"] is not None else None
        ),
        "gap_p95_p05": {
            "delta_1": stats_benign["p95"],
            "delta_2": stats_random["p05"],
            "gap": stats_random["p05"] - stats_benign["p95"],
            "gap_exists": stats_benign["p95"] < stats_random["p05"],
            "delta_1_source": "(A) paraphrase p95",
            "delta_2_source": "(B) random p05",
        },
        "_note": (
            "Gapped LSH is an embedding property. δ₁/δ₂ derived from "
            "(A) paraphrase + (B) random unrelated pairs. (C) adversarial "
            "stats included only for diagnostic — KCA source_prompt has a "
            "sim ∈ [0.5,0.75] filter, NDSS adv contains victim_query as "
            "substring; neither is a 'random unrelated query'."
        ),
    }

    # ---------- Print ----------
    print("\n" + "=" * 72)
    print(f"Assumption (i) Gapped LSH summary — {embedding}")
    print("=" * 72)
    for k, v in summary["stats"].items():
        if v is None:
            continue
        print(f"\n  {k}:")
        print(f"    n={v['n']}  mean={v['mean']:.4f}  std={v['std']:.4f}")
        print(f"    p05={v['p05']:.4f}  p25={v['p25']:.4f}  p50={v['p50']:.4f}  "
              f"p75={v['p75']:.4f}  p95={v['p95']:.4f}")

    print(f"\n  Percentile sweep (τ_dist = 1-τ_sim = {tau_dist:.3f}):")
    print(f"  {'α%':>4s} {'δ₁':>8s} {'δ₂':>8s} {'gap':>8s} "
          f"{'p₁':>6s} {'p₂':>6s} {'1-p₂':>8s} {'holds':>6s}")
    for r in sweep:
        p1 = f"{r['p_1']:.3f}" if r['p_1'] is not None else "  -  "
        p2 = f"{r['p_2']:.3f}" if r['p_2'] is not None else "  -  "
        det = (f"{r['detection_bound_1_minus_p2']:.3f}"
               if r['detection_bound_1_minus_p2'] is not None else "  -   ")
        h = "YES" if r["assumption_i_holds"] else "no"
        print(f"  {r['alpha_pct']:>4d} {r['delta_1']:>8.4f} {r['delta_2']:>8.4f} "
              f"{r['gap']:>+8.4f} {p1:>6s} {p2:>6s} {det:>8s} {h:>6s}")
    print(f"\n  critical α = {critical_alpha}%  "
          f"(smallest α where δ₁ < δ₂)")
    if headline.get("detection_bound_1_minus_p2") is not None:
        print(f"  headline (α=5%): main-theorem detection bound 1-p₂ = "
              f"{headline['detection_bound_1_minus_p2']:.3f}")

    # ---------- Save JSON / CSV ----------
    save_json(summary, out_dir / "summary.json")
    save_json({
        "embedding": embedding, "track": track,
        "tau_dist": tau_dist,
        "d_benign_paraphrase": d_benign.tolist(),
        "d_random_benign": d_random.tolist(),
        "d_adversarial": d_adv.tolist(),
    }, out_dir / "raw_data.json")
    df = pd.DataFrame({
        "type": (
            ["benign_paraphrase"] * len(d_benign)
            + ["random_benign"] * len(d_random)
            + ["adversarial"] * len(d_adv)
        ),
        "distance": list(d_benign) + list(d_random) + list(d_adv),
    })
    save_csv(df, out_dir / "distances.csv")
    save_csv(pd.DataFrame(sweep), out_dir / "gap_sweep.csv")

    if not HAVE_PLT:
        print("\n[plot] matplotlib not available; skipping figures")
        return

    # ---------- Plot 1: histogram with τ + δ₁ + δ₂ markers ----------
    # Headline LSH plot uses (A) paraphrase + (B) random ONLY.
    # (C) adversarial overlaid as faint dashed (diagnostic).
    fig, ax = plt.subplots(figsize=(10, 6))
    max_d = max(d_random.max(), d_adv.max() if len(d_adv) else 0) + 0.05
    bins = np.linspace(0, max_d, 60)
    ax.hist(d_benign, bins=bins, alpha=0.6, density=True, color="#2e8b57",
            label=f"(A) paraphrase  n={len(d_benign)}")
    ax.hist(d_random, bins=bins, alpha=0.55, density=True, color="#3b6fb5",
            label=f"(B) random unrelated  n={len(d_random)}")
    if len(d_adv):
        ax.hist(d_adv, bins=bins, alpha=0.25, density=True, color="#c0392b",
                histtype="step", linewidth=1.5,
                label=f"(C) adversarial (diagnostic only)  n={len(d_adv)}")
    ax.axvline(tau_dist, ls="--", color="black", lw=1.6,
               label=f"τ_dist = 1-τ_sim = {tau_dist:.3f}")
    ax.axvline(stats_benign["p95"], ls=":", color="#2e8b57", lw=1.6,
               label=f"δ₁ = (A) p95 = {stats_benign['p95']:.3f}")
    ax.axvline(stats_random["p05"], ls=":", color="#3b6fb5", lw=1.6,
               label=f"δ₂ = (B) p05 = {stats_random['p05']:.3f}")
    ax.set_xlabel("Cosine distance (1 - similarity)")
    ax.set_ylabel("Density")
    ax.set_title(f"exp2: Gapped LSH on {embedding}  (paraphrase vs random)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "histogram.png", dpi=150)
    plt.close()
    print(f"[saved] {out_dir / 'histogram.png'}")

    # ---------- Plot 2: gap sweep — δ₁, δ₂ vs α ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    alphas = [r["alpha_pct"] for r in sweep]
    d1s = [r["delta_1"] for r in sweep]
    d2s = [r["delta_2"] for r in sweep]
    ax.plot(alphas, d1s, "o-", color="#2e8b57", lw=2,
            label="δ₁ = (A) paraphrase p(100-α)")
    ax.plot(alphas, d2s, "s-", color="#3b6fb5", lw=2,
            label="δ₂ = (B) random p(α)")
    ax.fill_between(alphas, d1s, d2s,
                     where=[d1 < d2 for d1, d2 in zip(d1s, d2s)],
                     color="#2e8b57", alpha=0.15, label="gap region")
    ax.fill_between(alphas, d1s, d2s,
                     where=[d1 >= d2 for d1, d2 in zip(d1s, d2s)],
                     color="#c0392b", alpha=0.15, label="overlap region")
    ax.axhline(tau_dist, ls="--", color="black", alpha=0.6,
               label=f"τ_dist = {tau_dist:.3f}")
    if critical_alpha is not None:
        ax.axvline(critical_alpha, ls="--", color="purple", lw=1.8,
                   label=f"critical α = {critical_alpha}%")
    ax.set_xlabel("Tail mass α (%)")
    ax.set_ylabel("Distance")
    ax.set_title(f"exp2: Gap (δ₂ − δ₁) vs tail mass α — {track}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(out_dir / "gap_sweep.png", dpi=150)
    plt.close()
    print(f"[saved] {out_dir / 'gap_sweep.png'}")

    # ---------- Plot 3: tail probabilities (log scale) ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    sb = np.sort(d_benign); sr = np.sort(d_random)
    yb = np.arange(1, len(sb) + 1) / len(sb)
    yr = np.arange(1, len(sr) + 1) / len(sr)
    ax.plot(sb, 1 - yb, color="#2e8b57", lw=2,
            label="(A) paraphrase: P(d > x)")
    ax.plot(sr, yr, color="#3b6fb5", lw=2,
            label="(B) random unrelated: P(d < x)")
    ax.axvline(tau_dist, ls="--", color="black", alpha=0.6,
               label=f"τ_dist = {tau_dist:.3f}")
    ax.set_xlabel("Distance x")
    ax.set_ylabel("Tail probability")
    ax.set_title(f"exp2: Tail probabilities (log scale) — {embedding}")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "tails.png", dpi=150)
    plt.close()
    print(f"[saved] {out_dir / 'tails.png'}")


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--num-pairs", type=int, default=500,
                        help="pairs per type (A, B, C); default 500 to match "
                             "the rest of the v3.1 pipeline")
    parser.add_argument("--threshold", type=float, default=None,
                        help="τ_cache (sim space). Default: read from "
                             "outputs_<track>/calibration.json")
    parser.add_argument("--dataset-path", type=str, default="data/sc_ipi_v3.jsonl")
    parser.add_argument("--ndss-method", type=str,
                        default="prompt_injection_white_box",
                        choices=["zero_shot", "in_context_learning", "prompt_injection",
                                 "zero_shot_white_box", "in_context_learning_white_box",
                                 "prompt_injection_white_box"],
                        help="NDSS only: which method's adv to pair with victim")
    parser.add_argument("--quick", action="store_true",
                        help="smoke: 50/50/50 pairs")
    parser.add_argument("--output-prefix", type=str, default="exp2")
    args = parser.parse_args()

    if args.threshold is None:
        args.threshold = get_cache_threshold(args.embedding, args.track)
    if args.quick:
        args.num_pairs = 50

    run(track=args.track, num_pairs=args.num_pairs,
        embedding=args.embedding, threshold=args.threshold,
        ndss_method=args.ndss_method,
        seed=args.seed, dataset_path=args.dataset_path,
        output_prefix=args.output_prefix)


if __name__ == "__main__":
    main()
