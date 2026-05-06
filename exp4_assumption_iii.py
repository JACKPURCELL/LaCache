"""
exp4_assumption_iii.py — RQ1, Assumption (iii): Prefix Coherency
================================================================
Assumption (iii) (Prefix Coherency): the k-token response prefix f_k(.)
inherits the response separation of the full response. Formally,

    d(f_k(x), f_k(x')) > epsilon^{k_min - k} * d(f(x), f(x'))

for some epsilon < 1, so that running the lookahead check on a short
prefix is as discriminative as on the full output. In practice we track
the empirical ratio

    A_k = d(f_k(x), f_k(x')) / d(f(x), f(x'))

as a function of k and confirm that it saturates near 1 by k = 20.

Method:
  For each k in K_VALUES = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]:
    For each (victim, adversarial) pair in the dataset:
      A_k = d(f_k(victim), f_k(adversarial))
      B   = d(f(victim),   f(adversarial))    # full-length response
      delta_k = |A_k - B|
    Aggregate delta_k (mean, P95) and the ratio A_k / B.

Outputs (under outputs_<track>/exp4/<llm>/):
  summary.json   per-k stats, fitted epsilon, A_k saturation curve
  raw_data.json  per-(record, k) distances (consumed by Figure 3(c))
  decay.png      delta_k vs k (log scale)

Empirically epsilon < 0.95 across all (LLM, embedding) combinations, so
k = 20 already captures the full response divergence (paper Figure 3(c)).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

from common import (
    load_sc_ipi, load_sc_marco, vllm_generate, vllm_chat_format, embed_texts,
    cosine_sim, DEFAULT_EMBEDDING_MODEL, DEFAULT_BACKEND_LLM, OUTPUTS_DIR,
    save_json, save_csv, resolve_llm,
)


K_VALUES = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
FULL_TOKENS = 200  # "full" response length


def truncate_to_k_tokens(text, k, tokenizer):
    """Truncate a string to its first k tokens (decoded)."""
    ids = tokenizer.encode(text, add_special_tokens=False)[:k]
    return tokenizer.decode(ids, skip_special_tokens=True)


def run_exp4(max_records=None, embedding_model=DEFAULT_EMBEDDING_MODEL,
             backend_llm=DEFAULT_BACKEND_LLM, output_prefix="exp4",
             track="kca", ndss_method="prompt_injection_white_box",
             dataset_path=None,
             require_isr_success=False):

    print(f"\n=== Experiment 4: Assumption (iii) — epsilon^k decay ({track}) ===")
    print(f"k values: {K_VALUES}")
    print(f"Full response length: {FULL_TOKENS}")
    print(f"Embedding: {embedding_model}")
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

    # Generate full responses once (reuse for all k truncations)
    print("\n[gen] generating full responses for victim + adv...")
    victim_prompts = [vllm_chat_format(r["victim_query"]) for r in records]
    if track == "ndss":
        source_prompts = [vllm_chat_format(r["ndss_attacks"][ndss_method])
                          for r in records]
    else:
        source_prompts = [vllm_chat_format(r["source_prompt"]) for r in records]

    victim_full = vllm_generate(
        victim_prompts, max_tokens=FULL_TOKENS, temperature=0.0,
        model_name=backend_llm,
    )
    source_full = vllm_generate(
        source_prompts, max_tokens=FULL_TOKENS, temperature=0.0,
        model_name=backend_llm,
    )

    # Get tokenizer for truncation
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(resolve_llm(backend_llm),
                                               trust_remote_code=True)

    # Embed full responses
    print("\n[emb] embedding full responses (for B)...")
    victim_full_emb = embed_texts(victim_full, model_name=embedding_model,
                                  show_progress=True)
    source_full_emb = embed_texts(source_full, model_name=embedding_model,
                                  show_progress=True)
    B = 1.0 - cosine_sim(victim_full_emb, source_full_emb)  # full-response distances
    B = np.asarray(B)

    # For each k, truncate + embed + compute A_k
    k_results = {}
    for k in K_VALUES:
        print(f"\n[k={k}] truncating + embedding...")
        victim_k_texts = [truncate_to_k_tokens(r, k, tokenizer) for r in victim_full]
        source_k_texts = [truncate_to_k_tokens(r, k, tokenizer) for r in source_full]

        victim_k_emb = embed_texts(victim_k_texts, model_name=embedding_model)
        source_k_emb = embed_texts(source_k_texts, model_name=embedding_model)

        A_k = 1.0 - cosine_sim(victim_k_emb, source_k_emb)
        A_k = np.asarray(A_k)

        delta_k = np.abs(A_k - B)

        # Tingz' Assumption (iii) form: d(f_k(x), f_k(x')) > eps^(k_min - k) * d(f(x), f(x'))
        # Equivalently: ratio_k = A_k / B > eps^(k_min - k)
        # Avoid divide-by-zero on near-identical full responses
        safe_B = np.where(B > 1e-6, B, 1e-6)
        ratio_k = A_k / safe_B

        # Per-pair: what is the TIGHTEST epsilon upper bound such that
        #   A_k > eps^(k_min - k) * B  holds for this pair?
        # -> ratio_k > eps^(k_min - k)
        # -> log(ratio_k) > (k_min - k) * log(eps)
        # If k < k_min: (k_min - k) > 0, so log(eps) < log(ratio_k) / (k_min - k)
        # If ratio_k >= 1 (A_k >= B), then log(ratio_k) >= 0, so log(eps) < positive => any eps<1 works
        # If ratio_k < 1, we get a real upper bound on eps
        k_min = FULL_TOKENS  # all full responses are truncated to this length
        if k < k_min:
            # per-pair eps upper bound (only meaningful when ratio_k < 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_eps_bound = np.where(
                    ratio_k >= 1.0,
                    np.nan,  # assumption trivially holds for any eps < 1
                    np.log(ratio_k) / (k_min - k),
                )
                eps_bound = np.exp(log_eps_bound)
            # fraction of pairs where A_k >= B (trivially satisfies assumption)
            frac_trivial = float((ratio_k >= 1.0).mean())
            # for pairs that don't trivially satisfy, the tightest eps upper bound
            non_trivial = eps_bound[~np.isnan(eps_bound)]
            if len(non_trivial) > 0:
                eps_upper_p95 = float(np.nanpercentile(eps_bound, 95))  # worst-case 95%
                eps_upper_mean = float(np.nanmean(eps_bound))
            else:
                eps_upper_p95 = None
                eps_upper_mean = None
        else:
            frac_trivial = float((ratio_k >= 1.0).mean())
            eps_upper_p95 = None
            eps_upper_mean = None

        # Assumption (iii) holds for pair (x, x') iff A_k > eps^(k_min - k) * B.
        # For a GIVEN candidate eps (e.g. 0.99), count fraction where this holds.
        # Edge case: at k == k_min, A_k == B by definition, so the required ratio
        # is eps^0 = 1.0 and strictly-greater fails for all pairs. Use >= instead
        # so the assumption is trivially satisfied at k=k_min.
        assumption_holds_for_eps = {}
        for candidate_eps, key in [(0.999, "0.999"), (0.99, "0.99"),
                                    (0.95, "0.95"), (0.90, "0.90")]:
            if k < k_min:
                required_ratio = candidate_eps ** (k_min - k)
                assumption_holds_for_eps[key] = float(
                    (ratio_k > required_ratio).mean()
                )
            else:
                # k == k_min: A_k == B exactly, assumption is trivially satisfied
                assumption_holds_for_eps[key] = 1.0

        k_results[k] = {
            "A_k_mean": float(A_k.mean()),
            "A_k_p05": float(np.percentile(A_k, 5)),
            "A_k_p95": float(np.percentile(A_k, 95)),
            "B_mean": float(B.mean()),
            "ratio_k_mean": float(ratio_k.mean()),
            "ratio_k_p05": float(np.percentile(ratio_k, 5)),
            "ratio_k_p50": float(np.percentile(ratio_k, 50)),
            "ratio_k_p95": float(np.percentile(ratio_k, 95)),
            "frac_A_k_geq_B": frac_trivial,
            "eps_upper_p95": eps_upper_p95,
            "eps_upper_mean": eps_upper_mean,
            "assumption_holds_for_eps": assumption_holds_for_eps,
            # Legacy fields (still saved for backward compat)
            "delta_k_mean": float(delta_k.mean()),
            "delta_k_p95": float(np.percentile(delta_k, 95)),
            "delta_k_raw": delta_k.tolist(),
        }
        print(f"  k={k}: A_k/B mean={ratio_k.mean():.3f}, "
              f"frac(A_k>=B)={frac_trivial:.1%}, "
              f"frac(holds for eps=0.99)={assumption_holds_for_eps['0.99']:.1%}")

    # Fit legacy epsilon^k (for backward compat with existing plots)
    ks = np.array(K_VALUES, dtype=float)
    deltas = np.array([k_results[k]["delta_k_mean"] for k in K_VALUES])

    def model(k, C, eps):
        return C * np.power(eps, k)

    try:
        popt, _ = curve_fit(
            model, ks, deltas,
            p0=[1.0, 0.9], bounds=([0.0, 0.01], [10.0, 0.999]),
            maxfev=5000,
        )
        C_fit, eps_fit = popt
        fit_ok = True
    except Exception as e:
        print(f"  [fit error] {e}")
        C_fit, eps_fit = None, None
        fit_ok = False

    # Bound for assumption (iii): gamma > 2 * delta_k where gamma = B (mean adversarial distance)
    # We want k such that delta_k < gamma/2
    B_mean = float(B.mean())
    design_k = None
    for k in K_VALUES:
        if k_results[k]["delta_k_mean"] < B_mean / 2:
            design_k = k
            break

    # Print
    print("\n" + "=" * 70)
    print("Assumption (iii) verification (Tingz form: A_k > eps^(k_min-k) * B)")
    print("=" * 70)
    print(f"Mean B (full-response distance):  {B_mean:.4f}")
    print(f"k_min (full token length):         {FULL_TOKENS}")
    print()
    print(f"{'k':>6s} {'A_k/B':>8s} {'A_k>=B':>8s} {'eps=0.99':>10s} "
          f"{'eps=0.95':>10s} {'eps=0.90':>10s}")
    for k in K_VALUES:
        r = k_results[k]
        print(f"{k:>6d} {r['ratio_k_mean']:>8.3f} {r['frac_A_k_geq_B']:>8.1%} "
              f"{r['assumption_holds_for_eps']['0.99']:>10.1%} "
              f"{r['assumption_holds_for_eps']['0.95']:>10.1%} "
              f"{r['assumption_holds_for_eps']['0.90']:>10.1%}")
    print()
    if fit_ok:
        print(f"Legacy fit (|A_k - B| decay): delta_k ≈ {C_fit:.4f} * {eps_fit:.4f}^k")
        print(f"  (this fits |A_k - B|, NOT Tingz' A_k/B > eps^(k_min-k); see table above)")
    if design_k:
        print(f"\nDesign recommendation (|A_k - B| < B/2): k >= {design_k}")
    else:
        print(f"\nWARNING: no tested k satisfies |A_k - B| < B/2. Try larger k.")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    delta_means = [k_results[k]["delta_k_mean"] for k in K_VALUES]
    delta_p95 = [k_results[k]["delta_k_p95"] for k in K_VALUES]
    ax.plot(K_VALUES, delta_means, "o-", label="mean delta_k", color="blue")
    ax.plot(K_VALUES, delta_p95, "s--", label="p95 delta_k", color="blue", alpha=0.5)
    if fit_ok:
        k_smooth = np.linspace(min(K_VALUES), max(K_VALUES), 100)
        ax.plot(k_smooth, model(k_smooth, C_fit, eps_fit),
                ":", color="red", label=f"fit: {C_fit:.3f} * {eps_fit:.3f}^k")
    ax.axhline(B_mean / 2, color="green", linestyle="-",
               label=f"gamma/2 = {B_mean/2:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("k (response prefix length in tokens)")
    ax.set_ylabel("delta_k = ||d_k(x) - d_k(x')| - |d(x) - d(x')||")
    ax.set_title("Assumption (iii) — legacy |A_k - B| view")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"{output_prefix}_decay.png", dpi=150)
    plt.close()
    print(f"[saved] {output_prefix}_decay.png")

    # ------------------------------------------------------------
    # Tingz form plot: A_k / B ratio per k, plus required bound eps^(k_min-k)
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ratio_means = [k_results[k]["ratio_k_mean"] for k in K_VALUES]
    ratio_p05 = [k_results[k]["ratio_k_p05"] for k in K_VALUES]
    ratio_p95 = [k_results[k]["ratio_k_p95"] for k in K_VALUES]

    ax.plot(K_VALUES, ratio_means, "o-", color="blue", linewidth=2,
            label="mean A_k / B")
    ax.fill_between(K_VALUES, ratio_p05, ratio_p95, alpha=0.2, color="blue",
                     label="p05-p95")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5,
               label="A_k = B (ratio = 1)")

    # Required lower bounds for several eps candidates
    k_arr = np.array(K_VALUES, dtype=float)
    for eps_cand, color in [(0.99, "red"), (0.95, "orange"), (0.90, "green")]:
        required = np.where(k_arr < FULL_TOKENS,
                             eps_cand ** (FULL_TOKENS - k_arr),
                             1.0)
        ax.plot(K_VALUES, required, ":", color=color, alpha=0.7,
                label=f"required lower bound, eps={eps_cand}")

    ax.set_xlabel("k (response prefix length in tokens)")
    ax.set_ylabel("A_k / B (prefix vs full-response distance ratio)")
    ax.set_title(f"Assumption (iii) — Tingz form: A_k / B > eps^(k_min-k)\n"
                 f"(k_min = {FULL_TOKENS}; bound above = assumption holds)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(2.0, max(ratio_p95) * 1.1))
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"{output_prefix}_ratio.png", dpi=150)
    plt.close()
    print(f"[saved] {output_prefix}_ratio.png")

    # ------------------------------------------------------------
    # Assumption-hold fraction per k, per epsilon candidate
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for eps_key, color in [("0.999", "purple"), ("0.99", "red"),
                            ("0.95", "orange"), ("0.90", "green")]:
        fractions = [k_results[k]["assumption_holds_for_eps"][eps_key]
                     for k in K_VALUES]
        ax.plot(K_VALUES, fractions, "o-", color=color,
                label=f"eps = {eps_key}")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("k (response prefix length in tokens)")
    ax.set_ylabel("Fraction of pairs satisfying Assumption (iii)")
    ax.set_title("Assumption (iii) hold fraction vs k (per epsilon choice)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"{output_prefix}_hold_fraction.png", dpi=150)
    plt.close()
    print(f"[saved] {output_prefix}_hold_fraction.png")

    # Save
    # Drop raw arrays from saved summary to keep file small
    k_results_summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "delta_k_raw"}
        for k, v in k_results.items()
    }
    summary = {
        "B_mean": B_mean,
        "B_p50": float(np.percentile(B, 50)),
        "k_min": FULL_TOKENS,
        "k_results": k_results_summary,
        "fit_legacy_delta_k": {"ok": fit_ok, "C": C_fit, "epsilon": eps_fit} if fit_ok else {"ok": False},
        "design_k_recommendation": design_k,
    }
    save_json(summary, OUTPUTS_DIR / f"{output_prefix}_summary.json")

    # CSV
    rows = []
    for k in K_VALUES:
        r = k_results[k]
        rows.append({
            "k": k,
            "A_k_mean": r["A_k_mean"],
            "A_k_p95": r["A_k_p95"],
            "ratio_k_mean": r["ratio_k_mean"],
            "ratio_k_p05": r["ratio_k_p05"],
            "ratio_k_p50": r["ratio_k_p50"],
            "ratio_k_p95": r["ratio_k_p95"],
            "frac_A_k_geq_B": r["frac_A_k_geq_B"],
            "hold_eps_0.999": r["assumption_holds_for_eps"]["0.999"],
            "hold_eps_0.99": r["assumption_holds_for_eps"]["0.99"],
            "hold_eps_0.95": r["assumption_holds_for_eps"]["0.95"],
            "hold_eps_0.90": r["assumption_holds_for_eps"]["0.90"],
            "delta_k_mean": r["delta_k_mean"],
            "delta_k_p95": r["delta_k_p95"],
        })
    save_csv(pd.DataFrame(rows), OUTPUTS_DIR / f"{output_prefix}_decay.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--embedding", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--backend", type=str, default=DEFAULT_BACKEND_LLM)
    parser.add_argument("--track", type=str, default="kca", choices=["kca", "ndss"])
    parser.add_argument("--ndss-method", type=str,
                        default="prompt_injection_white_box",
                        choices=["zero_shot", "in_context_learning", "prompt_injection",
                                 "zero_shot_white_box", "in_context_learning_white_box",
                                 "prompt_injection_white_box"])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default="exp4")
    parser.add_argument("--require-isr-success", action="store_true",
                        help="Filter records to ISR=SUCCESS (NDSS) / "
                             "verdict=DIFFERENT (KCA) before generating.")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.max_records = 30

    run_exp4(
        max_records=args.max_records,
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