"""
gcg_perplexity_filter.py — distilGPT-2 perplexity helper for GCG soft loss.

Spec (KCA paper Eq. 3): GCG joint loss
    loss(adv) = (1 - sim(victim, adv)) + lambda * normalize_ppl(adv)
    normalize_ppl(p) = log(1+p) / log(1+ref)     (ref = 1500 by default)

Usage:
    from gcg_perplexity_filter import (
        ppl_of_text, ppl_of_texts, normalize_ppl, joint_loss,
    )
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

PPL_NORM_REF_DEFAULT = 1500.0
PPL_MODEL_NAME = "distilgpt2"

_PPL_MODEL = None
_PPL_TOKENIZER = None
_PPL_DEVICE = None


def _resolve_ppl_device():
    """Pick a device for distilGPT-2.

    Priority:
      1. LACACHE_PPL_DEVICE env var (e.g. "cuda:1", "cpu", "cuda")
      2. If multiple CUDA devices visible, use the LAST one (assumes the
         backbone LLM is on cuda:0)
      3. If exactly one CUDA device, use it
      4. CPU otherwise
    """
    import os
    import torch
    override = os.environ.get("LACACHE_PPL_DEVICE", "").strip()
    if override:
        return override
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n > 1:
            return f"cuda:{n - 1}"
        return "cuda:0"
    return "cpu"


def _load_ppl_model():
    global _PPL_MODEL, _PPL_TOKENIZER, _PPL_DEVICE
    if _PPL_MODEL is not None:
        return _PPL_MODEL, _PPL_TOKENIZER, _PPL_DEVICE
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _PPL_DEVICE = _resolve_ppl_device()
    print(f"[ppl] loading {PPL_MODEL_NAME} on {_PPL_DEVICE}")
    _PPL_TOKENIZER = AutoTokenizer.from_pretrained(PPL_MODEL_NAME)
    if _PPL_TOKENIZER.pad_token_id is None:
        _PPL_TOKENIZER.pad_token = _PPL_TOKENIZER.eos_token
    _PPL_MODEL = AutoModelForCausalLM.from_pretrained(PPL_MODEL_NAME).to(_PPL_DEVICE).eval()
    return _PPL_MODEL, _PPL_TOKENIZER, _PPL_DEVICE


def ppl_of_text(text: str, max_length: int = 256) -> float:
    """Per-token perplexity of `text` under distilGPT-2."""
    if not text or not text.strip():
        return float("inf")
    import torch
    model, tok, device = _load_ppl_model()
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc.input_ids.to(device)
        if ids.shape[1] < 2:
            return float("inf")
        out = model(ids, labels=ids)
        # out.loss is mean negative log-likelihood per token
        return float(math.exp(out.loss.item()))


def ppl_of_texts(texts: List[str], max_length: int = 256,
                 batch_size: int = 64) -> List[float]:
    """Batched per-token perplexity. Single forward per chunk, then per-row
    masked NLL using gather. Auto-halves batch_size on OOM."""
    if not texts:
        return []
    import os, torch
    model, tok, device = _load_ppl_model()
    # Allow env override of batch size (e.g. LACACHE_PPL_BATCH_SIZE=16)
    env_bs = os.environ.get("LACACHE_PPL_BATCH_SIZE", "").strip()
    if env_bs.isdigit():
        batch_size = max(1, int(env_bs))
    out: List[float] = [float("inf")] * len(texts)
    with torch.no_grad():
        start = 0
        while start < len(texts):
            cur_bs = batch_size
            chunk = texts[start:start + cur_bs]
            chunk = [t if t and t.strip() else " " for t in chunk]
            try:
                enc = tok(chunk, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(device)
                input_ids = enc.input_ids
                attn = enc.attention_mask
                # Single batched forward
                logits = model(input_ids, attention_mask=attn).logits
                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()       # (B, T-1, V)
                shift_labels = input_ids[:, 1:].contiguous()        # (B, T-1)
                shift_attn = attn[:, 1:].contiguous().float()       # (B, T-1)
                # log_softmax + gather (memory-efficient, no full V dim cross-entropy)
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                tgt_log_probs = log_probs.gather(
                    2, shift_labels.unsqueeze(-1)
                ).squeeze(-1)                                        # (B, T-1)
                tgt_log_probs = tgt_log_probs * shift_attn
                row_count = shift_attn.sum(dim=1).clamp(min=1.0)     # (B,)
                row_nll = -tgt_log_probs.sum(dim=1) / row_count
                row_count_int = shift_attn.sum(dim=1).long()
                for i, (loss, cnt) in enumerate(zip(row_nll.cpu().tolist(),
                                                      row_count_int.cpu().tolist())):
                    if cnt < 1:
                        continue
                    out[start + i] = math.exp(loss)
                start += len(chunk)
            except torch.cuda.OutOfMemoryError:
                if batch_size <= 1:
                    print(f"  [ppl OOM at batch_size=1] giving up on chunk; "
                          f"PPL stays inf for {len(chunk)} entries")
                    start += len(chunk)
                    continue
                old_bs = batch_size
                batch_size = max(1, batch_size // 2)
                print(f"  [ppl OOM at batch_size={old_bs}] retrying with "
                      f"batch_size={batch_size}")
                torch.cuda.empty_cache()
                # don't advance start, retry with smaller batch
    return out


def normalize_ppl(ppl: float, ref: float = PPL_NORM_REF_DEFAULT) -> float:
    """log(1+ppl) / log(1+ref). Maps ppl=ref to 1.0; ppl=0 to 0."""
    if ppl is None or not np.isfinite(ppl):
        return 1.0
    return math.log(1.0 + max(ppl, 0.0)) / math.log(1.0 + ref)


def joint_loss(sim: float, ppl: float, ppl_lambda: float = 0.3,
               ref: float = PPL_NORM_REF_DEFAULT) -> float:
    """KCA Eq. 3: (1 - sim) + lambda * normalize_ppl(ppl)."""
    return (1.0 - float(sim)) + ppl_lambda * normalize_ppl(ppl, ref)


def joint_loss_batch(sims: np.ndarray, ppls: List[float],
                     ppl_lambda: float = 0.3,
                     ref: float = PPL_NORM_REF_DEFAULT) -> np.ndarray:
    sims = np.asarray(sims, dtype=np.float64)
    norm_ppl = np.array([normalize_ppl(p, ref) for p in ppls], dtype=np.float64)
    return (1.0 - sims) + ppl_lambda * norm_ppl
