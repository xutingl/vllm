"""EditKV hidden state capture hooks for vLLM model runner.

Registers PyTorch forward pre-hooks on decoder layers to capture the
input hidden states during prefill. Supports two modes:

1. GPU-only: Store hidden states on GPU for GPU-side attention map computation.
2. CPU offload: Async-copy hidden states and K cache to pinned CPU memory
   via a dedicated CUDA stream, for CPU-side relevance dict computation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class EditKVHiddenStateCapture:
    """Captures per-layer input hidden states during prefill.

    When cpu_offload=True, also async-copies H and K to pinned CPU memory
    via a dedicated CUDA stream.

    Usage:
        capture = EditKVHiddenStateCapture(num_layers, cpu_offload=True)
        capture.register_hooks(model)
        # ... run forward pass ...
        hidden_states = capture.get_hidden_states()
        # For CPU offload:
        capture.synchronize_cpu()
        cpu_h, cpu_k = capture.get_cpu_data()
        capture.clear()
    """

    def __init__(self, num_layers: int, cpu_offload: bool = False):
        self.num_layers = num_layers
        self._hidden_states: list[torch.Tensor | None] = [None] * num_layers
        self._hooks: list[Any] = []
        self._enabled = False

        # CPU offload state
        self._cpu_offload = cpu_offload
        self._cpu_h: list[torch.Tensor | None] = [None] * num_layers
        self._cpu_k: list[torch.Tensor | None] = [None] * num_layers
        self._copy_stream: torch.cuda.Stream | None = None
        if cpu_offload:
            self._copy_stream = torch.cuda.Stream()

    def register_hooks(self, model: nn.Module) -> None:
        """Register forward pre-hooks on decoder layers.

        Looks for layers matching the pattern model.layers[i] (LlamaModel)
        or similar decoder layer containers.
        """
        layers = _find_decoder_layers(model)
        if len(layers) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} decoder layers, found {len(layers)}"
            )

        for layer_idx, layer in enumerate(layers):
            hook = layer.register_forward_pre_hook(
                self._make_hook(layer_idx), with_kwargs=True
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        """Create a forward pre-hook for a specific layer."""
        def hook(module, args, kwargs):
            if self._enabled:
                # LlamaDecoderLayer.forward(positions, hidden_states, residual)
                # hidden_states is args[1] (post-layernorm in some cases)
                hidden_states = args[1] if len(args) > 1 else None
                if hidden_states is not None:
                    self._hidden_states[layer_idx] = hidden_states.detach()
                    # Async copy to CPU if offloading
                    if self._cpu_offload and self._copy_stream is not None:
                        src = hidden_states.detach()
                        with torch.cuda.stream(self._copy_stream):
                            cpu_buf = torch.empty_like(src, device="cpu").pin_memory()
                            cpu_buf.copy_(src, non_blocking=True)
                            self._cpu_h[layer_idx] = cpu_buf
        return hook

    def capture_key_cache_for_layer(self, layer_idx: int, k_tensor: torch.Tensor) -> None:
        """Async-copy K cache for a layer to CPU (called after layer computes).

        Args:
            layer_idx: Layer index.
            k_tensor: K cache tensor on GPU.
        """
        if not self._cpu_offload or self._copy_stream is None:
            return
        with torch.cuda.stream(self._copy_stream):
            cpu_buf = torch.empty_like(k_tensor, device="cpu").pin_memory()
            cpu_buf.copy_(k_tensor.detach(), non_blocking=True)
            self._cpu_k[layer_idx] = cpu_buf

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def get_hidden_states(self) -> list[torch.Tensor | None]:
        return self._hidden_states

    def synchronize_cpu(self) -> None:
        """Wait for all async CPU copies to complete."""
        if self._copy_stream is not None:
            self._copy_stream.synchronize()

    def get_cpu_data(self) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
        """Return (cpu_h_list, cpu_k_list). Must call synchronize_cpu() first."""
        return self._cpu_h, self._cpu_k

    def clear(self) -> None:
        self._hidden_states = [None] * self.num_layers
        self._cpu_h = [None] * self.num_layers
        self._cpu_k = [None] * self.num_layers

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def _find_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Find decoder layers in a vLLM model.

    Supports LlamaModel and similar architectures where decoder layers
    are stored in model.model.layers or model.layers.
    """
    # Try model.model.layers (LlamaForCausalLM wraps LlamaModel)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    # Try model.layers directly
    if hasattr(model, 'layers'):
        return list(model.layers)
    # Try finding ModuleList children
    for child in model.children():
        if isinstance(child, nn.ModuleList):
            return list(child)
        if hasattr(child, 'layers') and isinstance(child.layers, nn.ModuleList):
            return list(child.layers)
    raise ValueError("Could not find decoder layers in model")


def compute_block_attention_scores(
    hidden_states: list[torch.Tensor],
    model: nn.Module,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    block_size: int = 16,
    top_k: int = 8,
    aggregation: str = "max",
) -> dict[int, torch.Tensor]:
    """Compute block-level attention scores from captured hidden states.

    For each layer, reconstructs Q from hidden_states using the model's
    QKV projection, then computes attention scores against K blocks
    from the KV cache.

    Args:
        hidden_states: Per-layer input hidden states [seq_len, hidden_dim].
        model: The vLLM model (for accessing QKV projection weights).
        kv_cache: Paged KV cache [num_blocks, 2, block_size, num_kv_heads, head_dim].
        block_table: Block indices for this request [num_blocks_for_req].
        seq_len: Actual sequence length.
        block_size: Tokens per block.
        top_k: Number of top attending blocks to store per key block.
        aggregation: "max" or "mean" for within-block score aggregation.

    Returns:
        Dict mapping layer_idx to tensor of shape [num_blocks, top_k] containing
        the top-k attending block indices for each key block.
    """
    layers = _find_decoder_layers(model)
    num_blocks = (seq_len + block_size - 1) // block_size
    block_scores = {}

    for layer_idx, layer in enumerate(layers):
        hs = hidden_states[layer_idx]
        if hs is None:
            continue

        attn = layer.self_attn
        hs_trimmed = hs[:seq_len]

        # Reconstruct Q: apply QKV projection and extract Q portion
        with torch.no_grad():
            qkv, _ = attn.qkv_proj(hs_trimmed)
            q = qkv[:, :attn.q_size]  # [seq_len, num_heads * head_dim]
            q = q.view(seq_len, attn.num_heads, attn.head_dim)

        # Read K from KV cache for this request's blocks
        k_blocks = []
        for block_idx in range(num_blocks):
            physical_page = block_table[block_idx].item()
            # kv_cache shape: [num_pages, 2, block_size, num_kv_heads, head_dim]
            k_block = kv_cache[physical_page, 0]  # [block_size, num_kv_heads, head_dim]
            tokens_in_block = min(block_size, seq_len - block_idx * block_size)
            k_blocks.append(k_block[:tokens_in_block])

        # Compute block-level attention scores
        # For each key block, compute aggregate attention from all query positions
        block_attn = torch.zeros(num_blocks, num_blocks, device=q.device)

        for kb_idx in range(num_blocks):
            k_block = k_blocks[kb_idx]  # [tokens_in_block, num_kv_heads, head_dim]
            # Expand K heads to match Q heads (GQA)
            num_groups = attn.num_heads // attn.num_kv_heads
            k_expanded = k_block.unsqueeze(2).expand(
                -1, -1, num_groups, -1
            ).reshape(-1, attn.num_heads, attn.head_dim)

            # Q @ K^T: [seq_len, num_heads, head_dim] @ [tokens_in_block, num_heads, head_dim]^T
            # -> [num_heads, seq_len, tokens_in_block]
            scores = torch.einsum('shd,thd->hst', q, k_expanded) * attn.scaling

            # Aggregate over heads and tokens within blocks
            # scores: [num_heads, seq_len, tokens_in_block]
            if aggregation == "max":
                # Max over tokens in key block, then mean over heads
                block_score = scores.max(dim=-1).values.mean(dim=0)  # [seq_len]
            else:
                block_score = scores.mean(dim=-1).mean(dim=0)  # [seq_len]

            # Aggregate query positions into query blocks
            for qb_idx in range(num_blocks):
                q_start = qb_idx * block_size
                q_end = min(q_start + block_size, seq_len)
                if aggregation == "max":
                    block_attn[qb_idx, kb_idx] = block_score[q_start:q_end].max()
                else:
                    block_attn[qb_idx, kb_idx] = block_score[q_start:q_end].mean()

        # Select top-k attending blocks for each key block
        k_actual = min(top_k, num_blocks)
        _, top_indices = block_attn.topk(k_actual, dim=0)  # [top_k, num_blocks]
        block_scores[layer_idx] = top_indices.T  # [num_blocks, top_k]

    return block_scores
