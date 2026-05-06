"""
gcg_attack.py — GCG optimizer for embedding-space cache collision
==================================================================
Implements the Greedy Coordinate Gradient (GCG) suffix/prefix optimizer
that backs both attack tracks in the paper:
  - KCA   (suffix-position, with optional `Neglect:` anchor)
  - SCP   (prefix-position, no anchor)

Loss (joint with perplexity-aware filter):
  L = (1 - sim(emb(adv_query), emb(target))) + lambda_ppl * log(1+ppl)/log(1+1500)
where lambda_ppl = 0.3 by default (paper appendix: GCG Hyperparameters).

Defaults match the paper's GCG configuration:
  variable_length = 20  optimizer_steps = 200  top_k_per_position = 64
  tau_tgt          = cache_tau + 0.03  (from common.get_dynamic_target_threshold)

Public API:
  construct_adv_query(source_prompt, suffix, neglect_prefix=True) -> str
  gcg_optimize(...)            -> (suffix, sim, steps)            (legacy tuple)
  gcg_optimize_full(...)       -> dict with full metadata          (main entry)
  gcg_optimize_adaptive(...)   -> (suffix, sim, steps)             (adaptive variant)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Make parent (LaCache/) importable so we can pull in common.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import get_embedding_model, cosine_sim
from gcg_perplexity_filter import (
    PPL_NORM_REF_DEFAULT, joint_loss_batch, ppl_of_text, ppl_of_texts,
)


GCG_MODES = ("random", "gradient")
NEGLECT_TOKEN = "Neglect:"


# =============================================================================
# adv_query construction
# =============================================================================

def construct_adv_query(source_prompt: str, suffix: str,
                         neglect_prefix: bool = True,
                         position: str = "suffix") -> str:
    """Build the full adv_query exactly as it'll be embedded by the cache.

    position="suffix" (KCA, default): "{source} [Neglect:] {suffix}"
        — GCG-optimized tokens go after the source_prompt.
    position="prefix" (NDSS white-box): "{suffix} [Neglect:] {source}"
        — GCG-optimized tokens go BEFORE the source_prompt (which is the
        prompt-injection payload). This pulls the embedding closer to
        victim_query without disturbing the trailing payload.
    """
    suffix = (suffix or "").strip()
    if position == "prefix":
        if neglect_prefix:
            return f"{suffix} {NEGLECT_TOKEN} {source_prompt}".strip()
        return f"{suffix} {source_prompt}".strip()
    # default: suffix at end (KCA-style)
    if neglect_prefix:
        return f"{source_prompt} {NEGLECT_TOKEN} {suffix}".strip()
    return f"{source_prompt} {suffix}".strip()


# =============================================================================
# Tokenizer / valid id list
# =============================================================================

def _valid_token_ids(tokenizer) -> List[int]:
    """Return token ids safe for GCG: exclude special tokens, control tokens,
    and pure-whitespace / empty surface forms."""
    bad = set()
    for attr in ("all_special_ids",):
        ids = getattr(tokenizer, attr, None)
        if ids:
            bad.update(int(x) for x in ids)
    # Try to filter pure-whitespace / empty pieces.
    valid: List[int] = []
    vocab_size = int(getattr(tokenizer, "vocab_size", None) or len(tokenizer.get_vocab()))
    for tid in range(vocab_size):
        if tid in bad:
            continue
        try:
            piece = tokenizer.decode([tid], skip_special_tokens=True)
        except Exception:
            continue
        if not piece or not piece.strip():
            continue
        # Exclude tokens that decode to a single non-printable.
        if all(not c.isprintable() for c in piece):
            continue
        valid.append(tid)
    if not valid:
        # Fallback: all non-special.
        valid = [i for i in range(vocab_size) if i not in bad]
    return valid


def _decode_suffix(token_ids: List[int], tokenizer) -> str:
    """Decode optimized id list back to natural-text suffix."""
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


# =============================================================================
# Random-search GCG with KCA Eq.3 joint loss
# =============================================================================

def _gcg_random(
    source_prompt: str,
    victim_query: str,
    embedding_model_name: str,
    *,
    y_star: Optional[str] = None,
    y_victim: Optional[str] = None,
    suffix_length: int = 20,
    max_steps: int = 500,
    top_k: int = 64,
    threshold: float = 0.85,
    seed: int = 42,
    verbose: bool = False,
    neglect_prefix: bool = True,
    ppl_filter: bool = True,
    ppl_lambda: float = 0.3,
    ppl_norm_ref: float = PPL_NORM_REF_DEFAULT,
    position: str = "suffix",
) -> Dict[str, Any]:
    model = get_embedding_model(embedding_model_name)
    tokenizer = model.tokenizer
    rng = random.Random(seed)
    valid_ids = _valid_token_ids(tokenizer)
    n_valid = len(valid_ids)

    target_text = victim_query if y_victim is None else f"{victim_query} {y_victim}"
    target_emb = model.encode(
        [target_text], normalize_embeddings=True, convert_to_numpy=True,
    )[0]

    def _build_text(suffix_ids: List[int]) -> str:
        suffix_str = _decode_suffix(suffix_ids, tokenizer)
        adv = construct_adv_query(source_prompt, suffix_str, neglect_prefix,
                                   position=position)
        if y_star:
            adv = f"{adv} {y_star}"
        return adv

    def _eval_batch(list_of_ids: List[List[int]]):
        texts = [_build_text(ids) for ids in list_of_ids]
        embs = model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True,
            batch_size=min(len(texts), 128), show_progress_bar=False,
        )
        sims = embs @ target_emb
        if ppl_filter:
            ppls = ppl_of_texts(texts)
            losses = joint_loss_batch(sims, ppls, ppl_lambda=ppl_lambda,
                                      ref=ppl_norm_ref)
        else:
            ppls = [float("nan")] * len(texts)
            losses = 1.0 - np.asarray(sims, dtype=np.float64)
        return np.asarray(sims, dtype=np.float64), np.asarray(ppls, dtype=np.float64), losses, texts

    # Initialize
    suffix_ids = [valid_ids[rng.randint(0, n_valid - 1)] for _ in range(suffix_length)]
    sims0, ppls0, losses0, texts0 = _eval_batch([suffix_ids])
    best_sim = float(sims0[0])
    best_ppl = float(ppls0[0])
    best_loss = float(losses0[0])
    best_ids = list(suffix_ids)
    best_text = texts0[0]

    sim_traj: List[float] = [best_sim]
    ppl_traj: List[float] = [best_ppl]
    loss_traj: List[float] = [best_loss]

    step = 0
    hit = best_sim >= threshold
    for step in range(max_steps):
        if hit:
            break
        pos = rng.randint(0, suffix_length - 1)
        candidate_tokens = [valid_ids[rng.randint(0, n_valid - 1)] for _ in range(top_k)]
        candidate_ids_list = []
        for tok in candidate_tokens:
            ids = list(best_ids)
            ids[pos] = tok
            candidate_ids_list.append(ids)
        sims, ppls, losses, texts = _eval_batch(candidate_ids_list)
        # Pick the candidate that minimizes joint loss
        best_idx = int(np.argmin(losses))
        if losses[best_idx] < best_loss - 1e-9:
            best_loss = float(losses[best_idx])
            best_sim = float(sims[best_idx])
            best_ppl = float(ppls[best_idx])
            best_ids = candidate_ids_list[best_idx]
            best_text = texts[best_idx]
            if best_sim >= threshold:
                hit = True
        sim_traj.append(best_sim)
        ppl_traj.append(best_ppl)
        loss_traj.append(best_loss)
        if verbose and step % 20 == 0:
            print(f"  [random] step {step}: sim={best_sim:.4f} ppl={best_ppl:.1f} "
                  f"loss={best_loss:.4f}", flush=True)

    suffix_text = _decode_suffix(best_ids, tokenizer)
    return {
        "suffix": suffix_text,
        "suffix_token_ids": list(best_ids),
        "adv_query": best_text,
        "final_sim": best_sim,
        "final_ppl": best_ppl,
        "final_loss": best_loss,
        "num_steps": step + 1,
        "hit": hit,
        "all_step_sims": sim_traj,
        "all_step_ppls": ppl_traj,
        "all_step_losses": loss_traj,
        "embedding_model_used": embedding_model_name,
        "ppl_filter": bool(ppl_filter),
        "ppl_lambda": ppl_lambda,
        "ppl_norm_ref": ppl_norm_ref,
        "target_threshold": threshold,
        "neglect_prefix": neglect_prefix,
        "mode": "random",
    }


# =============================================================================
# Gradient-mode GCG (kept for ablation; soft-PPL applied at re-rank stage)
# =============================================================================

def _get_torch_embedding_model(model_name: str):
    model = get_embedding_model(model_name)
    inner = model._first_module().auto_model
    tokenizer = model.tokenizer
    return inner, tokenizer, model


def _detect_pooling_mode(st_model):
    """Read the pooling type from a SentenceTransformer's Pooling module.
    Returns one of: 'mean', 'cls', 'lasttoken', 'max', 'mean_sqrt'.
    Handles both old (multiple bool attrs) and new (single string attr)
    sentence-transformers APIs. Falls back to 'mean'."""
    for mod in st_model.modules():
        if mod.__class__.__name__ != "Pooling":
            continue
        # New API (sentence-transformers ≥ ~3.x): single string `pooling_mode`
        pm = getattr(mod, "pooling_mode", None)
        if isinstance(pm, str):
            pm_l = pm.lower()
            if "lasttoken" in pm_l:
                return "lasttoken"
            if "cls" in pm_l:
                return "cls"
            if "max" in pm_l:
                return "max"
            if "sqrt" in pm_l:
                return "mean_sqrt"
            return "mean"
        # Old API: per-mode booleans
        if getattr(mod, "pooling_mode_lasttoken", False):
            return "lasttoken"
        if getattr(mod, "pooling_mode_cls_token", False):
            return "cls"
        if getattr(mod, "pooling_mode_max_tokens", False):
            return "max"
        if getattr(mod, "pooling_mode_mean_sqrt_len_tokens", False):
            return "mean_sqrt"
        if getattr(mod, "pooling_mode_mean_tokens", False):
            return "mean"
        # get_config_dict fallback
        try:
            cfg = mod.get_config_dict()
            pm = cfg.get("pooling_mode", "")
            if "lasttoken" in pm.lower():
                return "lasttoken"
            if "cls" in pm.lower():
                return "cls"
            if "max" in pm.lower():
                return "max"
        except Exception:
            pass
        break
    return "mean"


def _encode_with_embeddings(hf_model, input_embeds, attention_mask,
                              pooling: str = "mean"):
    """Differentiable forward + pooling.  `pooling` must match the embedding
    model's actual pooling (read via _detect_pooling_mode):
      mean       — BERT/MiniLM/E5     (mean over non-pad tokens)
      cls        — BGE / GTE / etc.    (first token)
      lasttoken  — Qwen3-Embedding    (last non-pad token, decoder-style)
      max        — rare, max over non-pad tokens
    Output is L2-normalized per row."""
    import torch
    outputs = hf_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state                           # (B, L, H)
    mask_f = attention_mask.float()                              # (B, L)

    if pooling == "lasttoken":
        # Index of last non-pad token per row
        last_idx = mask_f.sum(dim=1).long().clamp(min=1) - 1     # (B,)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        pooled = hidden[batch_idx, last_idx, :]                  # (B, H)
    elif pooling == "cls":
        pooled = hidden[:, 0, :]
    elif pooling == "max":
        # Mask out pad positions with very negative value before max
        masked = hidden.masked_fill(
            mask_f.unsqueeze(-1) == 0, torch.finfo(hidden.dtype).min,
        )
        pooled = masked.max(dim=1).values
    elif pooling == "mean_sqrt":
        sum_hidden = (hidden * mask_f.unsqueeze(-1)).sum(dim=1)
        lens = mask_f.sum(dim=1).clamp(min=1e-9)
        pooled = sum_hidden / torch.sqrt(lens).unsqueeze(-1)
    else:  # "mean" — default
        sum_hidden = (hidden * mask_f.unsqueeze(-1)).sum(dim=1)
        lens = mask_f.sum(dim=1).clamp(min=1e-9)
        pooled = sum_hidden / lens.unsqueeze(-1)

    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled


def _gcg_gradient(
    source_prompt: str,
    victim_query: str,
    embedding_model_name: str,
    *,
    y_star: Optional[str] = None,
    y_victim: Optional[str] = None,
    suffix_length: int = 20,
    max_steps: int = 500,
    top_k: int = 64,
    threshold: float = 0.85,
    seed: int = 42,
    verbose: bool = False,
    neglect_prefix: bool = True,
    ppl_filter: bool = True,
    ppl_lambda: float = 0.3,
    ppl_norm_ref: float = PPL_NORM_REF_DEFAULT,
    position: str = "suffix",
) -> Dict[str, Any]:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_model, tokenizer, st_model = _get_torch_embedding_model(embedding_model_name)
    hf_model = hf_model.to(device).eval()
    valid_ids = _valid_token_ids(tokenizer)
    valid_id_t = torch.tensor(valid_ids, device=device)

    # Read the embedding model's actual pooling so sims match what
    # SentenceTransformer.encode() (and downstream cache layer) compute.
    pooling = _detect_pooling_mode(st_model)
    if verbose:
        print(f"  [gradient] embedding pooling mode: {pooling}", flush=True)

    embed_layer = hf_model.get_input_embeddings()
    rng = random.Random(seed)

    target_text = victim_query if y_victim is None else f"{victim_query} {y_victim}"
    target_enc = tokenizer(target_text, return_tensors="pt", truncation=True,
                            max_length=128).to(device)
    with torch.no_grad():
        target_emb = _encode_with_embeddings(
            hf_model,
            embed_layer(target_enc.input_ids),
            target_enc.attention_mask,
            pooling=pooling,
        )

    # Build full token list:
    # [CLS] source_tokens [Neglect:] suffix_slots [y_star] [SEP]
    src_tokens = tokenizer(source_prompt, add_special_tokens=False).input_ids
    neg_tokens = tokenizer(NEGLECT_TOKEN, add_special_tokens=False).input_ids if neglect_prefix else []
    y_tokens = tokenizer(y_star, add_special_tokens=False).input_ids if y_star else []
    cls = [tokenizer.cls_token_id] if tokenizer.cls_token_id is not None else []
    sep = [tokenizer.sep_token_id] if tokenizer.sep_token_id is not None else []

    suffix_ids = [valid_ids[rng.randint(0, len(valid_ids) - 1)] for _ in range(suffix_length)]

    if position == "prefix":
        # NDSS white-box: [CLS] suffix NEGLECT src_tokens y_tokens [SEP]
        def _full_input_ids(suffix: List[int]) -> List[int]:
            return cls + suffix + neg_tokens + src_tokens + y_tokens + sep
    else:
        # KCA default: [CLS] src_tokens NEGLECT suffix y_tokens [SEP]
        def _full_input_ids(suffix: List[int]) -> List[int]:
            return cls + src_tokens + neg_tokens + suffix + y_tokens + sep

    def _adv_text(suffix: List[int]) -> str:
        suffix_str = _decode_suffix(suffix, tokenizer)
        adv = construct_adv_query(source_prompt, suffix_str, neglect_prefix,
                                   position=position)
        if y_star:
            adv = f"{adv} {y_star}"
        return adv

    def _eval_one(suffix: List[int]):
        ids = torch.tensor([_full_input_ids(suffix)], device=device)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            emb = _encode_with_embeddings(hf_model, embed_layer(ids), mask,
                                          pooling=pooling)
            s = float(torch.nn.functional.cosine_similarity(emb, target_emb).item())
        ppl_v = ppl_of_text(_adv_text(suffix)) if ppl_filter else float("nan")
        return s, ppl_v

    best_sim, best_ppl = _eval_one(suffix_ids)
    best_loss = (1.0 - best_sim) + (ppl_lambda * (np.log(1 + best_ppl) / np.log(1 + ppl_norm_ref))
                                     if ppl_filter and np.isfinite(best_ppl) else 0.0)
    best_ids = list(suffix_ids)
    sim_traj = [best_sim]
    ppl_traj = [best_ppl]
    loss_traj = [best_loss]

    if position == "prefix":
        suffix_start = len(cls)
    else:
        suffix_start = len(cls) + len(src_tokens) + len(neg_tokens)
    vocab_size = embed_layer.weight.shape[0]
    step = 0
    hit = best_sim >= threshold
    for step in range(max_steps):
        if hit:
            break
        full_ids = _full_input_ids(best_ids)
        ids_t = torch.tensor([full_ids], device=device)
        mask = torch.ones_like(ids_t)
        one_hot = torch.zeros(suffix_length, vocab_size, device=device,
                              dtype=embed_layer.weight.dtype)
        for i, tok in enumerate(best_ids):
            one_hot[i, tok] = 1.0
        one_hot.requires_grad_()
        emb_mat = embed_layer(ids_t).detach().clone()
        suffix_embed = one_hot @ embed_layer.weight
        emb_mat[0, suffix_start: suffix_start + suffix_length, :] = suffix_embed
        pooled = _encode_with_embeddings(hf_model, emb_mat, mask, pooling=pooling)
        sim = torch.nn.functional.cosine_similarity(pooled, target_emb).sum()
        grad = torch.autograd.grad(sim, one_hot)[0]
        # Restrict candidates to valid_ids
        grad_at_valid = grad[:, valid_id_t]
        pos = rng.randint(0, suffix_length - 1)
        _, top_local = torch.topk(grad_at_valid[pos], top_k)
        top_tokens = [int(valid_id_t[i].item()) for i in top_local.tolist()]

        best_cand_loss = best_loss
        best_cand_sim = best_sim
        best_cand_ppl = best_ppl
        best_cand_tok = None
        for tok in top_tokens:
            if tok == best_ids[pos]:
                continue
            ids = list(best_ids)
            ids[pos] = tok
            s, p = _eval_one(ids)
            l = (1.0 - s) + (ppl_lambda * (np.log(1 + p) / np.log(1 + ppl_norm_ref))
                              if ppl_filter and np.isfinite(p) else 0.0)
            if l < best_cand_loss - 1e-9:
                best_cand_loss = l
                best_cand_sim = s
                best_cand_ppl = p
                best_cand_tok = tok

        if best_cand_tok is not None:
            best_ids[pos] = best_cand_tok
            best_loss = float(best_cand_loss)
            best_sim = float(best_cand_sim)
            best_ppl = float(best_cand_ppl)
            if best_sim >= threshold:
                hit = True
        sim_traj.append(best_sim)
        ppl_traj.append(best_ppl)
        loss_traj.append(best_loss)
        if verbose and step % 20 == 0:
            print(f"  [gradient] step {step}: sim={best_sim:.4f} ppl={best_ppl:.1f} "
                  f"loss={best_loss:.4f}", flush=True)

    suffix_text = _decode_suffix(best_ids, tokenizer)
    adv_query_text = _adv_text(best_ids)
    return {
        "suffix": suffix_text,
        "suffix_token_ids": list(best_ids),
        "adv_query": adv_query_text,
        "final_sim": best_sim,
        "final_ppl": best_ppl,
        "final_loss": best_loss,
        "num_steps": step + 1,
        "hit": hit,
        "all_step_sims": sim_traj,
        "all_step_ppls": ppl_traj,
        "all_step_losses": loss_traj,
        "embedding_model_used": embedding_model_name,
        "ppl_filter": bool(ppl_filter),
        "ppl_lambda": ppl_lambda,
        "ppl_norm_ref": ppl_norm_ref,
        "target_threshold": threshold,
        "neglect_prefix": neglect_prefix,
        "mode": "gradient",
    }


# =============================================================================
# Public API
# =============================================================================

def gcg_optimize_full(
    source_prompt: str,
    victim_query: str,
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    mode: str = "random",
    suffix_length: int = 20,
    max_steps: int = 500,
    top_k: int = 64,
    threshold: float = 0.85,
    seed: int = 42,
    verbose: bool = False,
    neglect_prefix: bool = True,
    ppl_filter: bool = True,
    ppl_lambda: float = 0.3,
    ppl_norm_ref: float = PPL_NORM_REF_DEFAULT,
    y_star: Optional[str] = None,
    y_victim: Optional[str] = None,
    position: str = "suffix",
) -> Dict[str, Any]:
    """v3.1 main API: returns the full result dict (suffix, sim, ppl, traj, metadata).
    `position`: "suffix" (KCA, default) or "prefix" (NDSS white-box)."""
    if mode == "random":
        impl = _gcg_random
    elif mode == "gradient":
        impl = _gcg_gradient
    else:
        raise ValueError(f"Unknown GCG mode: {mode!r}; expected one of {GCG_MODES}")
    return impl(
        source_prompt, victim_query, embedding_model_name,
        y_star=y_star, y_victim=y_victim,
        suffix_length=suffix_length, max_steps=max_steps, top_k=top_k,
        threshold=threshold, seed=seed, verbose=verbose,
        neglect_prefix=neglect_prefix,
        ppl_filter=ppl_filter, ppl_lambda=ppl_lambda, ppl_norm_ref=ppl_norm_ref,
        position=position,
    )


def gcg_optimize(
    source_prompt: str,
    victim_query: str,
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    mode: str = "random",
    suffix_length: int = 20,
    max_steps: int = 500,
    top_k: int = 64,
    threshold: float = 0.85,
    seed: int = 42,
    verbose: bool = False,
    neglect_prefix: bool = True,
    ppl_filter: bool = True,
    ppl_lambda: float = 0.3,
) -> Tuple[str, float, int]:
    """Legacy API: returns (suffix, final_sim, num_steps)."""
    res = gcg_optimize_full(
        source_prompt, victim_query, embedding_model_name,
        mode=mode, suffix_length=suffix_length, max_steps=max_steps,
        top_k=top_k, threshold=threshold, seed=seed, verbose=verbose,
        neglect_prefix=neglect_prefix, ppl_filter=ppl_filter,
        ppl_lambda=ppl_lambda,
    )
    return res["suffix"], res["final_sim"], res["num_steps"]


def gcg_optimize_batch_full(
    records: List[Dict[str, Any]],
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    *,
    suffix_length: int = 20,
    max_steps: int = 200,
    top_k: int = 64,
    threshold: float = 0.85,
    seed_base: int = 42,
    neglect_prefix: bool = True,
    ppl_filter: bool = True,
    ppl_lambda: float = 0.3,
    ppl_norm_ref: float = PPL_NORM_REF_DEFAULT,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Batched random-search GCG: run all `records` in lockstep.

    Each step builds (n_active * top_k) candidates across records and
    evaluates them in a single embedding forward + single PPL forward.
    Records that hit `threshold` drop out of the active set early.

    `records` must be a list of dicts with at least keys:
        source_prompt, victim_query
    The returned list is ordered to match `records`.
    """
    import numpy as np
    import math

    model = get_embedding_model(embedding_model_name)
    tokenizer = model.tokenizer
    valid_ids = _valid_token_ids(tokenizer)
    n_valid = len(valid_ids)
    n = len(records)
    if n == 0:
        return []

    # Per-record state
    rngs = [random.Random(seed_base + i) for i in range(n)]
    target_texts = [r["victim_query"] for r in records]
    target_embs = model.encode(
        target_texts, normalize_embeddings=True, convert_to_numpy=True,
        batch_size=min(len(target_texts), 256), show_progress_bar=False,
    )

    state = []
    for i, r in enumerate(records):
        suffix_ids = [valid_ids[rngs[i].randint(0, n_valid - 1)] for _ in range(suffix_length)]
        state.append({
            "source_prompt": r["source_prompt"],
            "suffix_ids": suffix_ids,
            "best_sim": -1.0,
            "best_ppl": float("inf"),
            "best_loss": float("inf"),
            "best_text": "",
            "hit": False,
            "active": True,
            "sim_traj": [],
            "ppl_traj": [],
            "loss_traj": [],
        })

    def _build_text(source_prompt: str, suffix_ids: List[int]) -> str:
        suffix_str = _decode_suffix(suffix_ids, tokenizer)
        return construct_adv_query(source_prompt, suffix_str, neglect_prefix)

    def _norm_ppl(p: float) -> float:
        if not np.isfinite(p):
            return 1.0
        return math.log(1.0 + max(p, 0.0)) / math.log(1.0 + ppl_norm_ref)

    # Initial eval (one candidate per record = its random init)
    init_texts = [_build_text(s["source_prompt"], s["suffix_ids"]) for s in state]
    init_embs = model.encode(
        init_texts, normalize_embeddings=True, convert_to_numpy=True,
        batch_size=min(len(init_texts), 256), show_progress_bar=False,
    )
    init_ppls = ppl_of_texts(init_texts) if ppl_filter else [float("nan")] * n
    for i, s in enumerate(state):
        sim = float(init_embs[i] @ target_embs[i])
        ppl = init_ppls[i]
        loss = (1.0 - sim) + (ppl_lambda * _norm_ppl(ppl) if ppl_filter else 0.0)
        s["best_sim"] = sim
        s["best_ppl"] = ppl
        s["best_loss"] = loss
        s["best_text"] = init_texts[i]
        s["sim_traj"].append(sim)
        s["ppl_traj"].append(ppl)
        s["loss_traj"].append(loss)
        if sim >= threshold:
            s["hit"] = True
            s["active"] = False

    for step in range(max_steps):
        active_idxs = [i for i, s in enumerate(state) if s["active"]]
        if not active_idxs:
            break

        # Each active record proposes top_k candidates
        all_cands: List[Tuple[int, List[int]]] = []  # (record_idx, suffix_ids)
        for i in active_idxs:
            s = state[i]
            rng = rngs[i]
            pos = rng.randint(0, suffix_length - 1)
            for _ in range(top_k):
                tok_id = valid_ids[rng.randint(0, n_valid - 1)]
                cand = list(s["suffix_ids"])
                cand[pos] = tok_id
                all_cands.append((i, cand))

        texts = [_build_text(state[r]["source_prompt"], cand) for r, cand in all_cands]
        embs = model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True,
            batch_size=min(len(texts), 512), show_progress_bar=False,
        )
        ppls = ppl_of_texts(texts) if ppl_filter else [float("nan")] * len(texts)

        # Group candidate indices by record
        rec_to_cidx: Dict[int, List[int]] = {}
        for cidx, (r, _) in enumerate(all_cands):
            rec_to_cidx.setdefault(r, []).append(cidx)

        for r, cidxs in rec_to_cidx.items():
            s = state[r]
            cidxs_arr = np.asarray(cidxs)
            cand_sims = embs[cidxs_arr] @ target_embs[r]                 # (top_k,)
            cand_ppls = np.asarray([ppls[c] for c in cidxs])
            if ppl_filter:
                norm_ppls = np.array([_norm_ppl(p) for p in cand_ppls])
                cand_losses = (1.0 - cand_sims) + ppl_lambda * norm_ppls
            else:
                cand_losses = 1.0 - cand_sims
            best_local = int(np.argmin(cand_losses))
            if cand_losses[best_local] < s["best_loss"] - 1e-9:
                best_global_cidx = cidxs[best_local]
                s["best_sim"] = float(cand_sims[best_local])
                s["best_ppl"] = float(cand_ppls[best_local])
                s["best_loss"] = float(cand_losses[best_local])
                s["suffix_ids"] = list(all_cands[best_global_cidx][1])
                s["best_text"] = texts[best_global_cidx]
                if s["best_sim"] >= threshold:
                    s["hit"] = True
                    s["active"] = False
            s["sim_traj"].append(s["best_sim"])
            s["ppl_traj"].append(s["best_ppl"])
            s["loss_traj"].append(s["best_loss"])

        if verbose and step % 10 == 0:
            n_hit = sum(1 for s in state if s["hit"])
            avg_sim = float(np.mean([s["best_sim"] for s in state]))
            print(f"  [batch step {step:3d}]  active={len(active_idxs):3d}/{n}  "
                  f"hit={n_hit:3d}  avg_sim={avg_sim:.3f}", flush=True)

    # Pack results in input order
    results: List[Dict[str, Any]] = []
    for i, s in enumerate(state):
        suffix_text = _decode_suffix(s["suffix_ids"], tokenizer)
        results.append({
            "suffix": suffix_text,
            "suffix_token_ids": list(s["suffix_ids"]),
            "adv_query": s["best_text"] or _build_text(s["source_prompt"], s["suffix_ids"]),
            "final_sim": s["best_sim"],
            "final_ppl": s["best_ppl"],
            "final_loss": s["best_loss"],
            "num_steps": len(s["sim_traj"]),
            "hit": s["hit"],
            "all_step_sims": s["sim_traj"],
            "all_step_ppls": s["ppl_traj"],
            "all_step_losses": s["loss_traj"],
            "embedding_model_used": embedding_model_name,
            "ppl_filter": bool(ppl_filter),
            "ppl_lambda": ppl_lambda,
            "ppl_norm_ref": ppl_norm_ref,
            "target_threshold": threshold,
            "neglect_prefix": neglect_prefix,
            "mode": "random_batch",
        })
    return results


def gcg_optimize_adaptive(
    source_prompt: str,
    victim_query: str,
    y_star: str,
    y_victim: str,
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    mode: str = "random",
    suffix_length: int = 20,
    max_steps: int = 500,
    top_k: int = 64,
    threshold: float = 0.85,
    seed: int = 42,
    verbose: bool = False,
    neglect_prefix: bool = True,
    ppl_filter: bool = True,
    ppl_lambda: float = 0.3,
) -> Tuple[str, float, int]:
    """Adaptive variant: optimize sim(emb(adv ⊕ y_star), emb(victim ⊕ y_victim))."""
    res = gcg_optimize_full(
        source_prompt, victim_query, embedding_model_name,
        mode=mode, suffix_length=suffix_length, max_steps=max_steps,
        top_k=top_k, threshold=threshold, seed=seed, verbose=verbose,
        neglect_prefix=neglect_prefix, ppl_filter=ppl_filter,
        ppl_lambda=ppl_lambda,
        y_star=y_star, y_victim=y_victim,
    )
    return res["suffix"], res["final_sim"], res["num_steps"]
