#!/usr/bin/env bash
# run_all.sh — Run the full LaCache pipeline end-to-end.
#
# Pipeline:
#   (optional) (re)build attack dataset via GCG  -> build_dataset/build_*.py
#   (optional) refresh benign paraphrase pool    -> prepare_benign_pairs.py
#               run RQ1 Assumption (i)            -> exp2_lsh_calibration.py
#               run RQ1 Assumption (ii)           -> exp3_assumption_ii.py
#               run RQ1 Assumption (iii)          -> exp4_assumption_iii.py
#               run RQ2 LaCache vs. baselines    -> exp5_lacache_defense.py

set -euo pipefail
cd "$(dirname "$0")"

# =============================================================================
# Configuration (edit these)
# =============================================================================

DATASET="scp"                                            # kca | scp
LLM="qwen"                                                # qwen | gemma
EMBEDDING="BAAI/bge-large-en-v1.5"                        # BAAI/bge-large-en-v1.5 | sentence-transformers/all-MiniLM-L6-v2 | Qwen/Qwen3-Embedding-0.6B

# (Re)run GCG to refresh the white-box adversarial queries before running
# experiments. KCA -> rebuild the entire dataset (suffix-GCG is the whole
# attack). SCP -> rerun only the white-box prefix-GCG (black-box variants
# are template-based and ship pre-built in data/scp.jsonl).
# Slow: GCG is several GPU-hours per dataset.
RUN_GCG=false
GCG_WORKERS=4                                              # number of parallel shards (one per GPU)
GCG_STEPS=200                                              # GCG optimizer steps per record
GCG_BATCH_SIZE=16                                          # GCG batch size (KCA builder only)
GCG_PPL_LAMBDA=0.3                                         # GCG perplexity-aware filter weight (paper default)

# Regenerate the benign paraphrase pool via GPT calls.
# Required only if benign_pairs.jsonl is missing or you want a fresh pool.
REFRESH_BENIGN_PAIRS=false
NUM_BENIGN_PAIRS=500                                        # size of benign pool

# Per-experiment subset / hyperparameters.
LIMIT=500                                                   # records per experiment (use 50 for a quick smoke)
NUM_BENIGN=200                                              # benign pairs used by exp5 BHR
K=20                                                        # lookahead window length

# SCP-only: which SCP variant to evaluate against.
# zero_shot | in_context_learning | prompt_injection
# (or *_white_box for GCG-prefixed variants)
SCP_METHOD="prompt_injection_white_box"

# =============================================================================
# Derived
# =============================================================================

# Internal track name (paper "SCP" == code "ndss"). Map here so the rest of
# the script speaks paper terminology while invoking the unchanged CLIs.
case "$DATASET" in
    kca) TRACK="kca" ;;
    scp) TRACK="ndss" ;;
    *)   echo "DATASET must be one of: kca, scp"; exit 1 ;;
esac

DATASET_PATH=""
case "$DATASET" in
    kca) DATASET_PATH="data/sc_ipi_v3.jsonl" ;;
    scp) DATASET_PATH="data/scp.jsonl" ;;
esac
case "$EMBEDDING" in
    *MiniLM*) DATASET_PATH="${DATASET_PATH%.jsonl}.minilm.jsonl" ;;
esac

LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)_${DATASET}_${LLM}"
mkdir -p "$LOG_DIR"

EMB_SHORT="${EMBEDDING##*/}"
echo "============================================================"
echo "  LaCache pipeline"
echo "------------------------------------------------------------"
echo "  dataset           = $DATASET (track=$TRACK, file=$DATASET_PATH)"
echo "  LLM               = $LLM"
echo "  embedding         = $EMB_SHORT"
echo "  re-run GCG        = $RUN_GCG  (workers=$GCG_WORKERS, steps=$GCG_STEPS)"
echo "  refresh benign    = $REFRESH_BENIGN_PAIRS  (n=$NUM_BENIGN_PAIRS)"
echo "  limit / k         = $LIMIT / $K"
echo "  SCP variant       = $SCP_METHOD"
echo "  logs              = $LOG_DIR/"
echo "============================================================"

# =============================================================================
# (Optional) Stage 0: rebuild attack dataset via GCG
# =============================================================================

if [[ "$RUN_GCG" == "true" ]]; then
    echo
    echo "[stage 0] Rebuilding $DATASET dataset via GCG  (workers=$GCG_WORKERS)"
    case "$DATASET" in
        kca)
            # KCA: full rebuild (suffix-GCG is the entire attack pipeline).
            python build_dataset/build_sc_ipi_v3.py \
                --num-nq 100 --pairs-per-nq 5 --level 2 \
                --llm "$LLM" --embedding "$EMBEDDING" \
                --workers "$GCG_WORKERS" \
                --gcg-max-steps "$GCG_STEPS" \
                --gcg-batch-size "$GCG_BATCH_SIZE" \
                --ppl-lambda "$GCG_PPL_LAMBDA" \
                --output "$DATASET_PATH" \
                2>&1 | tee "$LOG_DIR/build_kca.log"
            ;;
        scp)
            # SCP: black-box variants ship pre-built; only refresh the
            # prefix-GCG white-box variants by sharding across GCG_WORKERS GPUs.
            SHARD_DIR="$LOG_DIR/scp_shards"
            mkdir -p "$SHARD_DIR"
            echo "[stage 0/scp] launching $GCG_WORKERS GCG shards (one per GPU)"
            for ((i=0; i<GCG_WORKERS; i++)); do
                CUDA_VISIBLE_DEVICES=$i python build_dataset/gcg_shard_ndss.py \
                    --shard-id $i --num-shards "$GCG_WORKERS" \
                    --dataset-path "$DATASET_PATH" \
                    --embedding "$EMBEDDING" \
                    --gcg-mode gradient --gcg-steps "$GCG_STEPS" \
                    --suffix-len 20 --top-k 64 \
                    --ppl-lambda "$GCG_PPL_LAMBDA" \
                    --ndss-method prompt_injection \
                    --output "$SHARD_DIR/scp.shard$i.jsonl" \
                    > "$LOG_DIR/build_scp_shard$i.log" 2>&1 &
            done
            wait
            echo "[stage 0/scp] merging shards back into $DATASET_PATH"
            python build_dataset/merge_gcg_shards.py \
                --original "$DATASET_PATH" \
                --shards "$SHARD_DIR/scp.shard*.jsonl" \
                --output "$DATASET_PATH" \
                2>&1 | tee "$LOG_DIR/merge_scp.log"
            ;;
    esac
fi

# =============================================================================
# (Optional) Stage 0b: refresh benign paraphrase pool
# =============================================================================

if [[ "$REFRESH_BENIGN_PAIRS" == "true" ]]; then
    echo
    echo "[stage 0b] Generating $NUM_BENIGN_PAIRS benign paraphrase pairs"
    python prepare_benign_pairs.py --num "$NUM_BENIGN_PAIRS" \
        2>&1 | tee "$LOG_DIR/benign_pairs.log"
fi

# =============================================================================
# Stage 1: RQ1 — empirical validation of the three working assumptions
# =============================================================================

echo
echo "[stage 1.1] Assumption (i) Gapped LSH"
python exp2_lsh_calibration.py \
    --track "$TRACK" --embedding "$EMBEDDING" \
    --num-pairs "$LIMIT" \
    --dataset-path "$DATASET_PATH" \
    2>&1 | tee "$LOG_DIR/exp2.log"

echo
echo "[stage 1.2] Assumption (ii) Semantic Separation"
python exp3_assumption_ii.py \
    --track "$TRACK" --backend "$LLM" --embedding "$EMBEDDING" \
    --max-records "$LIMIT" \
    --dataset-path "$DATASET_PATH" \
    2>&1 | tee "$LOG_DIR/exp3.log"

echo
echo "[stage 1.3] Assumption (iii) Prefix Coherency"
python exp4_assumption_iii.py \
    --track "$TRACK" --backend "$LLM" --embedding "$EMBEDDING" \
    --max-records "$LIMIT" \
    --dataset-path "$DATASET_PATH" \
    2>&1 | tee "$LOG_DIR/exp4.log"

# =============================================================================
# Stage 2: RQ2 — LaCache effectiveness vs. baselines
# =============================================================================

echo
echo "[stage 2] LaCache vs. baseline defenses"
python exp5_lacache_defense.py \
    --track "$TRACK" --llm "$LLM" --embedding "$EMBEDDING" \
    --limit "$LIMIT" --num-benign "$NUM_BENIGN" --k "$K" \
    --ndss-method "$SCP_METHOD" \
    --dataset-path "$DATASET_PATH" \
    2>&1 | tee "$LOG_DIR/exp5.log"

echo
echo "All stages completed. Logs at $LOG_DIR/.  Outputs at outputs_${TRACK}/."
