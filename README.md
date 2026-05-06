# LaCache

![LaCache overview](LaCache.png)

Code and datasets for the paper **"LaCache: Lookahead Semantic Caching for Robust LLM Serving"**.

LaCache is a lightweight extension of semantic caching that augments the cache-key check on the query with a **lookahead check** on the first *k* tokens of the LLM response. A cache hit is served only when both checks pass; otherwise the request falls back to full decoding. This release contains the code that reproduces the assumption-validation experiments (RQ1) and the main defense experiment (RQ2) reported in the paper.

## Repository layout

```
LaCache/
├── README.md                       this file
├── common.py                       shared utilities (loaders, vLLM, embeddings, judges, thresholds)
├── prepare_benign_pairs.py         build the benign paraphrase pool
├── exp2_lsh_calibration.py         RQ1, Assumption (i): Gapped LSH
├── exp3_assumption_ii.py           RQ1, Assumption (ii): Semantic Separation
├── exp4_assumption_iii.py          RQ1, Assumption (iii): Prefix Coherency
├── exp5_lacache_defense.py         RQ2: LaCache vs. paraphrase-recheck baseline (Tables 1, 2)
├── run_all.sh                      end-to-end pipeline runner (configurable header)
├── build_dataset/
│   ├── build_sc_ipi_v3.py          KCA dataset builder (Natural Questions, suffix-GCG)
│   ├── build_sc_marco_v1.py        SCP dataset builder (MS-MARCO, black-box variants)
│   ├── gcg_shard_ndss.py           SCP white-box GCG (prefix mode, sharded)
│   ├── merge_gcg_shards.py         merge per-shard outputs back into the canonical dataset
│   ├── gcg_attack.py               GCG suffix/prefix optimizer (used by builders/shards)
│   └── gcg_perplexity_filter.py    distilGPT-2 perplexity helper used inside GCG
└── data/
    ├── sc_ipi_v3.jsonl             KCA dataset, BGE-embedded (paper default)
    ├── sc_ipi_v3.minilm.jsonl      KCA dataset, MiniLM-embedded
    ├── scp.jsonl                   SCP dataset, BGE-embedded (paper default)
    ├── scp.minilm.jsonl            SCP dataset, MiniLM-embedded
    └── benign_pairs.jsonl          500 benign paraphrase pairs (BHR pool)
```

GCG attack code lives under `build_dataset/` because it is only invoked when (re)building the attack datasets; the four shipped experiments (`exp2`–`exp5`) run on the pre-built JSONL files in `data/` and never call GCG at runtime.

## Code-to-paper notation map

The released code carries some legacy internal names. The mapping below is exhaustive; everything else matches the paper notation directly.

| Code symbol                | Paper symbol      | Meaning                                                         |
|----------------------------|-------------------|-----------------------------------------------------------------|
| `track == "kca"`           | KCA               | Key-Collision Attack track                                       |
| `track == "ndss"`          | SCP               | Semantic Cache Poisoning track                                   |
| `sc_ipi_v3.jsonl`          | KCA dataset       | 500-record KCA evaluation set                                    |
| `scp.jsonl`                | SCP dataset       | 500-record SCP evaluation set                                    |
| `cache_threshold` / `tau_c`| τ_c               | Cache admission threshold (5th percentile of benign sim, capped 0.80) |
| `guard_threshold` / `tau_g`| τ_a               | Lookahead admission threshold (10th percentile of benign response sim, capped 0.70) |
| `tau_tgt` / `target_threshold` | τ_tgt         | GCG early-stop similarity (= τ_c + 0.03)                          |
| `k`                        | k                 | Lookahead window length (default 20)                              |
| `r_sim`                    | sim(f_k(x), f_k(x'))  | Lookahead similarity                                          |
| `q_sim`                    | sim(x, x')        | Cache-key similarity                                              |
| `isr_verdict == "SUCCESS"` | ISR=Success       | Subset on which AHR is reported                                   |

## Setup

Python 3.12; tested on H100. We recommend a fresh venv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Set the OpenAI API key (used for paraphrase generation, ISR judging, dataset rebuilding):

```bash
export OPENAI_API_KEY=sk-...
```

The pre-built datasets ship under `data/`; **you do not need to rebuild them** to reproduce RQ1 / RQ2.

## End-to-end runner

The simplest way to reproduce a single configuration is `run_all.sh`. Edit the configuration block at the top of the script (dataset, LLM, embedding, whether to rebuild the dataset via GCG, GCG worker count, ...) and then:

```bash
bash run_all.sh
```

It will sequentially run (optional GCG rebuild) → (optional benign pool refresh) → exp2 → exp3 → exp4 → exp5 and write all logs to `logs/<timestamp>/`.

## Reproducing RQ1: Working assumptions (paper Figure 3)

```bash
# Assumption (i) Gapped LSH — measures benign / random / adversarial distance distributions
python exp2_lsh_calibration.py --track kca  --embedding BAAI/bge-large-en-v1.5
python exp2_lsh_calibration.py --track ndss --embedding BAAI/bge-large-en-v1.5

# Assumption (ii) Semantic Separation — measures d(f(x*), f(x)) on ISR=SUCCESS subset
python exp3_assumption_ii.py --track kca  --llm qwen   --embedding BAAI/bge-large-en-v1.5
python exp3_assumption_ii.py --track ndss --llm gemma  --embedding sentence-transformers/all-MiniLM-L6-v2

# Assumption (iii) Prefix Coherency — A_k saturation at k = 20
python exp4_assumption_iii.py --track ndss --llm qwen --embedding BAAI/bge-large-en-v1.5
```

Each script writes `summary.json` and `raw_data.json` to `outputs_<track>/exp{2,3,4}/` for downstream plotting.

## Reproducing RQ2: LaCache vs. baselines (paper Table 1, Table 2)

```bash
# KCA track on the default Qwen3 + BGE pair
python exp5_lacache_defense.py --track kca  --llm qwen --embedding BAAI/bge-large-en-v1.5 --k 20

# SCP track, all four (LLM, embedding) combinations
python exp5_lacache_defense.py --track ndss --llm qwen  --embedding BAAI/bge-large-en-v1.5 --k 20
python exp5_lacache_defense.py --track ndss --llm qwen  --embedding sentence-transformers/all-MiniLM-L6-v2 --k 20
python exp5_lacache_defense.py --track ndss --llm gemma --embedding BAAI/bge-large-en-v1.5 --k 20
python exp5_lacache_defense.py --track ndss --llm gemma --embedding sentence-transformers/all-MiniLM-L6-v2 --k 20
```

Each run writes `summary.json` (AHR / BHR per defense) and `raw_data.json` to `outputs_<track>/exp5/<llm>/`.

## Threshold calibration

Both `τ_c` and `τ_a` are read from a per-track calibration file `outputs_<track>/calibration.json`. If the file is absent, the scripts fall back to the global defaults `τ_c = 0.78`, `τ_a = 0.70`. You can regenerate the calibration file from `exp2_lsh_calibration.py` outputs (see paper appendix "Threshold Calibration" for the exact formula).

## (Optional) Rebuilding the datasets

The shipped datasets in `data/` are sufficient to reproduce all numbers in the paper. The simplest way to refresh the GCG attacks while keeping everything else fixed is to set `RUN_GCG=true` in `run_all.sh` and rerun it.

For finer control:

```bash
# KCA dataset (Natural Questions + suffix-GCG, full rebuild incl. ISR judge)
python build_dataset/build_sc_ipi_v3.py \
    --num-nq 100 --pairs-per-nq 5 --level 2 \
    --embedding BAAI/bge-large-en-v1.5 --llm qwen \
    --output data/sc_ipi_v3.jsonl

# SCP dataset (MS-MARCO + black-box template variants; GPT-heavy, only needed
# for a full from-scratch rebuild of data/scp.jsonl)
python build_dataset/build_sc_marco_v1.py \
    --target-total 500 --llm qwen \
    --output data/scp.jsonl

# SCP white-box (prefix-GCG only) — sharded for parallelism
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python build_dataset/gcg_shard_ndss.py \
      --shard-id $i --num-shards 4 \
      --dataset-path data/scp.jsonl --embedding BAAI/bge-large-en-v1.5 \
      --gcg-mode gradient --gcg-steps 200 --suffix-len 20 \
      --output tmp_shards/scp.shard$i.jsonl &
done
wait
python build_dataset/merge_gcg_shards.py \
    --original data/scp.jsonl --shards "tmp_shards/scp.shard*.jsonl" \
    --output data/scp.jsonl

# Benign paraphrase pool
python prepare_benign_pairs.py --num 500
```

KCA's builder is stage-checkpointed; reruns resume from the last completed stage. The SCP black-box generation in `build_sc_marco_v1.py` is template-based and does NOT call GCG, so it does not need refreshing when only the embedding model changes — re-running `gcg_shard_ndss.py` is sufficient.

## Hardware

Each experiment fits on `4 x H100`. Full RQ1 + RQ2 reproduction takes ~3 GPU-hours.
