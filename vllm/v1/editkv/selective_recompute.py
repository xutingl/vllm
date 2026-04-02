"""Selective block recompute for EditKV using FlashInfer paged attention.

Given a set of selected block indices, runs a partial forward pass that:
1. Only processes tokens from selected blocks through the model
2. Writes new K/V only for selected block slots
3. Reads the full KV cache during attention (selected tokens attend to all context)

This achieves compute savings of (1 - recompute_ratio) * suffix_length
compared to vLLM's full suffix recompute with prefix caching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper


def compute_selective_inputs(
    edited_input_ids: list[int],
    selected_block_indices: list[int],
    block_table: np.ndarray,
    block_size: int,
    seq_len: int,
) -> dict[str, np.ndarray | torch.Tensor]:
    """Prepare inputs for a selective recompute forward pass.

    Args:
        edited_input_ids: Full token IDs for the edited document.
        selected_block_indices: Block indices to recompute (sorted).
        block_table: Physical page IDs for this request, shape [max_blocks].
        block_size: Tokens per block.
        seq_len: Total sequence length.

    Returns:
        Dict with:
            "input_ids": Token IDs for selected positions only. [num_selected_tokens]
            "positions": Original position IDs (for RoPE). [num_selected_tokens]
            "selective_slot_mapping": Cache slots to write to. [num_selected_tokens]
            "all_page_indices": All physical pages for attention read. [num_blocks]
            "num_selected_tokens": Number of tokens being processed.
    """
    num_blocks = (seq_len + block_size - 1) // block_size

    # Expand selected blocks to token positions
    selected_positions = []
    for b in selected_block_indices:
        start = b * block_size
        end = min(start + block_size, seq_len)
        selected_positions.extend(range(start, end))

    selected_positions = np.array(selected_positions, dtype=np.int64)
    num_selected = len(selected_positions)

    # Extract input_ids for selected positions
    ids_array = np.array(edited_input_ids, dtype=np.int64)
    selected_input_ids = ids_array[selected_positions]

    # Compute slot mapping for selected positions only
    block_indices = selected_positions // block_size
    block_offsets = selected_positions % block_size
    physical_pages = block_table[block_indices]
    selective_slot_mapping = physical_pages * block_size + block_offsets

    # All pages for attention read (full context)
    all_page_indices = block_table[:num_blocks].copy()

    return {
        "input_ids": torch.from_numpy(selected_input_ids),
        "positions": torch.from_numpy(selected_positions.copy()),
        "selective_slot_mapping": torch.from_numpy(selective_slot_mapping),
        "all_page_indices": torch.from_numpy(all_page_indices),
        "num_selected_tokens": num_selected,
        "num_total_blocks": num_blocks,
    }


def plan_selective_prefill(
    wrapper: "BatchPrefillWithPagedKVCacheWrapper",
    num_selected_tokens: int,
    all_page_indices: torch.Tensor,
    num_total_blocks: int,
    seq_len: int,
    block_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_data_type: torch.dtype = torch.bfloat16,
) -> None:
    """Plan the FlashInfer prefill for selective recompute.

    The key asymmetry:
    - Query side: only selected tokens (small)
    - KV side: all pages (full context)

    Args:
        wrapper: FlashInfer BatchPrefillWithPagedKVCacheWrapper instance.
        num_selected_tokens: Number of tokens being processed (query side).
        all_page_indices: Physical page IDs for ALL pages (KV read side).
        num_total_blocks: Total number of blocks in the sequence.
        seq_len: Full sequence length.
        block_size: Tokens per block.
        num_qo_heads: Number of query/output heads.
        num_kv_heads: Number of KV heads.
        head_dim: Head dimension.
        q_data_type: Query data type.
    """
    # Single request: qo_indptr = [0, num_selected_tokens]
    qo_indptr = torch.tensor([0, num_selected_tokens], dtype=torch.int32)

    # KV side: all pages for full context read
    paged_kv_indptr = torch.tensor([0, num_total_blocks], dtype=torch.int32)
    paged_kv_indices = all_page_indices.to(dtype=torch.int32)

    # Last page length
    last_page_len = seq_len - (num_total_blocks - 1) * block_size
    paged_kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32)

    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        block_size,
        causal=False,  # We handle causality via custom position mapping
        q_data_type=q_data_type,
    )


def selective_recompute_flashinfer(
    model: torch.nn.Module,
    kv_cache: list[torch.Tensor],
    edited_input_ids: list[int],
    selected_block_indices: list[int],
    block_table: np.ndarray,
    block_size: int,
    seq_len: int,
    device: str = "cuda",
) -> dict[str, int | float]:
    """Run selective recompute using FlashInfer paged attention.

    This is the vLLM-native selective recompute path. Instead of processing
    the entire edited sequence, it only forwards tokens from selected blocks
    through the model, writing K/V only for those blocks while reading the
    full KV cache during attention.

    Args:
        model: vLLM model (with model.model.layers, etc.).
        kv_cache: Per-layer paged KV cache tensors.
            Each tensor has shape [num_pages, 2, block_size, num_kv_heads, head_dim].
        edited_input_ids: Full token IDs for the edited document.
        selected_block_indices: Sorted list of block indices to recompute.
        block_table: Physical page IDs, shape [max_blocks_per_req].
        block_size: Tokens per block.
        seq_len: Total sequence length.
        device: CUDA device.

    Returns:
        Dict with statistics: num_selected_tokens, num_total_tokens, etc.
    """
    inputs = compute_selective_inputs(
        edited_input_ids, selected_block_indices, block_table, block_size, seq_len
    )

    num_selected = inputs["num_selected_tokens"]
    input_ids = inputs["input_ids"].to(device)
    positions = inputs["positions"].to(device)
    slot_mapping = inputs["selective_slot_mapping"].to(device)
    all_pages = inputs["all_page_indices"].to(device)

    # Get model structure
    base_model = model.model if hasattr(model, "model") else model
    embed_tokens = base_model.embed_tokens
    layers = base_model.layers

    # Embedding
    hidden_states = embed_tokens(input_ids.unsqueeze(0))  # [1, num_selected, hidden_dim]

    # RoPE embeddings at original positions
    position_ids = positions.unsqueeze(0)  # [1, num_selected]
    if hasattr(base_model, "rotary_emb"):
        cos, sin = base_model.rotary_emb(hidden_states, position_ids)
    else:
        cos, sin = None, None

    # Forward through decoder layers
    for layer_idx, decoder_layer in enumerate(layers):
        layer_kv = kv_cache[layer_idx]  # [num_pages, 2, block_size, num_kv_heads, head_dim]
        hidden_states = _selective_layer_forward(
            decoder_layer, hidden_states, positions, cos, sin,
            layer_kv, slot_mapping, all_pages, inputs["num_total_blocks"],
            seq_len, block_size, device,
        )

    return {
        "num_selected_tokens": num_selected,
        "num_total_tokens": seq_len,
        "num_selected_blocks": len(selected_block_indices),
        "num_total_blocks": inputs["num_total_blocks"],
        "compute_ratio": num_selected / seq_len,
    }


def _selective_layer_forward(
    decoder_layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    all_page_indices: torch.Tensor,
    num_total_blocks: int,
    seq_len: int,
    block_size: int,
    device: str,
) -> torch.Tensor:
    """Forward pass through a single decoder layer with selective KV update.

    Uses eager attention (matmul-based) as fallback. The FlashInfer kernel
    path would replace the attention computation for production use.
    """
    import torch.nn.functional as F
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    attn = decoder_layer.self_attn
    head_dim = attn.head_dim
    num_heads = attn.config.num_attention_heads
    num_kv_heads = attn.config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    num_selected = hidden_states.shape[1]

    # Pre-attention layernorm
    residual = hidden_states
    normed = decoder_layer.input_layernorm(hidden_states)

    # QKV projection
    query_states = attn.q_proj(normed).view(1, num_selected, num_heads, head_dim).transpose(1, 2)
    key_states = attn.k_proj(normed).view(1, num_selected, num_kv_heads, head_dim).transpose(1, 2)
    value_states = attn.v_proj(normed).view(1, num_selected, num_kv_heads, head_dim).transpose(1, 2)

    # Apply RoPE
    if cos is not None and sin is not None:
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Write new K/V to cache at selected slots
    _write_kv_to_paged_cache(key_states, value_states, kv_cache, slot_mapping, block_size)

    # Read full K/V from cache (all pages)
    K_full, V_full = _read_full_kv_from_paged_cache(
        kv_cache, all_page_indices, num_total_blocks, seq_len, block_size
    )
    # K_full, V_full: [1, num_kv_heads, seq_len, head_dim]

    # GQA expansion
    if num_kv_groups > 1:
        K_full = K_full.repeat_interleave(num_kv_groups, dim=1)
        V_full = V_full.repeat_interleave(num_kv_groups, dim=1)

    # Compute attention: selected queries attend to full context
    attn_weights = torch.matmul(query_states, K_full.transpose(-2, -1)) * attn.scaling
    # attn_weights: [1, num_heads, num_selected, seq_len]

    # Build causal mask: position i can only attend to positions <= positions[i]
    pos_expanded = positions.unsqueeze(-1)  # [num_selected, 1]
    all_positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
    causal_mask = (all_positions > pos_expanded).unsqueeze(0).unsqueeze(0)  # [1, 1, num_selected, seq_len]
    attn_weights.masked_fill_(causal_mask, float("-inf"))

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
    attn_output = torch.matmul(attn_weights, V_full)
    # attn_output: [1, num_heads, num_selected, head_dim]

    attn_output = attn_output.transpose(1, 2).reshape(1, num_selected, -1).contiguous()
    attn_output = attn.o_proj(attn_output)

    # Post-attention residual + MLP
    hidden_states = residual + attn_output
    residual = hidden_states
    normed = decoder_layer.post_attention_layernorm(hidden_states)
    hidden_states = decoder_layer.mlp(normed)
    hidden_states = residual + hidden_states

    return hidden_states


def _write_kv_to_paged_cache(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Write K/V to paged cache at specified slots.

    Args:
        key_states: [1, num_kv_heads, num_tokens, head_dim]
        value_states: [1, num_kv_heads, num_tokens, head_dim]
        kv_cache: [num_pages, 2, block_size, num_kv_heads, head_dim]
        slot_mapping: [num_tokens] mapping token → cache slot
        block_size: Tokens per block.
    """
    num_tokens = slot_mapping.shape[0]
    page_indices = slot_mapping // block_size
    offsets = slot_mapping % block_size

    # key_states: [1, num_kv_heads, num_tokens, head_dim] → [num_tokens, num_kv_heads, head_dim]
    k = key_states.squeeze(0).transpose(0, 1)  # [num_tokens, num_kv_heads, head_dim]
    v = value_states.squeeze(0).transpose(0, 1)  # [num_tokens, num_kv_heads, head_dim]

    for i in range(num_tokens):
        page = page_indices[i]
        offset = offsets[i]
        kv_cache[page, 0, offset] = k[i]  # K
        kv_cache[page, 1, offset] = v[i]  # V


def _read_full_kv_from_paged_cache(
    kv_cache: torch.Tensor,
    page_indices: torch.Tensor,
    num_blocks: int,
    seq_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read full K/V from paged cache for all blocks.

    Args:
        kv_cache: [num_pages, 2, block_size, num_kv_heads, head_dim]
        page_indices: Physical page IDs for this request, [num_blocks].
        num_blocks: Number of blocks.
        seq_len: Total sequence length.
        block_size: Tokens per block.

    Returns:
        K, V tensors of shape [1, num_kv_heads, seq_len, head_dim].
    """
    num_kv_heads = kv_cache.shape[3]
    head_dim = kv_cache.shape[4]

    K_parts = []
    V_parts = []
    for b in range(num_blocks):
        page = page_indices[b]
        tokens_in_block = min(block_size, seq_len - b * block_size)
        K_parts.append(kv_cache[page, 0, :tokens_in_block])  # [tokens, num_kv_heads, head_dim]
        V_parts.append(kv_cache[page, 1, :tokens_in_block])

    K_full = torch.cat(K_parts, dim=0).unsqueeze(0).permute(0, 2, 1, 3)  # [1, num_kv_heads, seq_len, head_dim]
    V_full = torch.cat(V_parts, dim=0).unsqueeze(0).permute(0, 2, 1, 3)

    return K_full, V_full
