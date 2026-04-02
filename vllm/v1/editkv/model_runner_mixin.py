"""EditKV mixin for GPUModelRunner.

Provides EditKV functionality that can be mixed into the model runner
without deeply modifying the core vLLM code. Handles:
1. Hidden state capture hook registration after model loading
2. Block attention score computation after prefill (GPU or CPU offload)
3. Selective recompute scheduling for edit requests
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from vllm.v1.editkv.config import EditKVConfig
from vllm.v1.editkv.hooks import EditKVHiddenStateCapture, compute_block_attention_scores
from vllm.v1.editkv.manager import EditKVManager

if TYPE_CHECKING:
    import torch.nn as nn


class EditKVModelRunnerMixin:
    """Mixin that adds EditKV support to GPUModelRunner."""

    def _editkv_init(self) -> None:
        """Initialize EditKV state. Call after __init__."""
        editkv_enabled = os.environ.get("VLLM_EDITKV_ENABLED", "0") == "1"
        if editkv_enabled:
            cpu_offload = os.environ.get("VLLM_EDITKV_CPU_OFFLOAD", "1") == "1"
            self.editkv_config = EditKVConfig(
                enabled=True,
                storage_backend=os.environ.get("VLLM_EDITKV_STORAGE", "gpu"),
                top_k_blocks=int(os.environ.get("VLLM_EDITKV_TOP_K", "8")),
                recompute_ratio=float(os.environ.get("VLLM_EDITKV_RECOMPUTE_RATIO", "0.2")),
                block_relevance_agg=os.environ.get("VLLM_EDITKV_AGG", "max"),
                cpu_offload_enabled=cpu_offload,
                cpu_relevance_threshold=float(
                    os.environ.get("VLLM_EDITKV_CPU_THRESHOLD", "0.01")
                ),
                cpu_fallback_ratio=float(
                    os.environ.get("VLLM_EDITKV_CPU_FALLBACK_RATIO", "0.5")
                ),
                cpu_num_threads=int(os.environ.get("VLLM_EDITKV_CPU_THREADS", "4")),
            )
            self.editkv_manager = EditKVManager(self.editkv_config)
            self.editkv_capture: EditKVHiddenStateCapture | None = None
        else:
            self.editkv_config = EditKVConfig(enabled=False)
            self.editkv_manager = None
            self.editkv_capture = None

    def _editkv_setup_hooks(self, model: nn.Module, num_layers: int) -> None:
        """Register hidden state capture hooks on the model.

        Call after load_model().
        """
        if not self.editkv_config.enabled:
            return

        cpu_offload = self.editkv_config.cpu_offload_enabled
        self.editkv_capture = EditKVHiddenStateCapture(
            num_layers, cpu_offload=cpu_offload
        )
        self.editkv_capture.register_hooks(model)

        # Initialize CPU relevance computer if using CPU offload
        if cpu_offload and self.editkv_manager is not None:
            self.editkv_manager.init_cpu_computer(model)

    def _editkv_on_prefill_start(self) -> None:
        """Called before a prefill forward pass to enable capture."""
        if self.editkv_capture is not None:
            self.editkv_capture.enable()

    def _editkv_on_prefill_end(
        self,
        request_id: str,
        input_ids: list[int],
        model: nn.Module,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
    ) -> None:
        """Called after a prefill forward pass to compute and store attention scores.

        For CPU offload: synchronizes async copies and submits background CPU computation.
        For GPU path: computes block attention scores on GPU (legacy).

        Args:
            request_id: Request identifier.
            input_ids: Token IDs of the prefilled sequence.
            model: The model (for Q projection weights).
            kv_cache: The paged KV cache tensor.
            block_table: Block indices for this request.
            seq_len: Sequence length.
        """
        if self.editkv_capture is None or self.editkv_manager is None:
            return

        self.editkv_capture.disable()

        # --- CPU offload path ---
        if self.editkv_config.cpu_offload_enabled:
            # Sync async copies and collect CPU data
            self.editkv_capture.synchronize_cpu()
            cpu_h, cpu_k = self.editkv_capture.get_cpu_data()

            # Also capture K from paged KV cache for layers that need it
            block_size = kv_cache.shape[2]
            num_blocks = (seq_len + block_size - 1) // block_size
            for layer_idx in range(self.editkv_capture.num_layers):
                if cpu_k[layer_idx] is None:
                    # Read K from paged cache and copy to CPU
                    k_blocks = []
                    for b in range(num_blocks):
                        page = block_table[b].item()
                        k_block = kv_cache[page, 0]  # [block_size, num_kv_heads, head_dim]
                        tokens = min(block_size, seq_len - b * block_size)
                        k_blocks.append(k_block[:tokens])
                    k_full = torch.cat(k_blocks, dim=0)  # [seq_len, num_kv_heads, head_dim]
                    self.editkv_capture.capture_key_cache_for_layer(layer_idx, k_full)

            self.editkv_capture.synchronize_cpu()
            cpu_h, cpu_k = self.editkv_capture.get_cpu_data()

            # Submit background CPU computation
            self.editkv_manager.register_prefill_cpu_offload(
                request_id=request_id,
                input_ids=input_ids,
                cpu_h=cpu_h,
                cpu_k=cpu_k,
                block_table=block_table.tolist(),
                seq_len=seq_len,
            )
            self.editkv_capture.clear()
            return

        # --- GPU path (legacy) ---
        hidden_states = self.editkv_capture.get_hidden_states()

        if not self.editkv_config.capture_attention_scores:
            self.editkv_manager.register_prefill(
                request_id=request_id,
                input_ids=input_ids,
                block_attention_scores=None,
                block_table=block_table.tolist(),
                seq_len=seq_len,
            )
            self.editkv_capture.clear()
            return

        block_size = kv_cache.shape[2]
        block_scores = compute_block_attention_scores(
            hidden_states=hidden_states,
            model=model,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_len=seq_len,
            block_size=block_size,
            top_k=self.editkv_config.top_k_blocks,
            aggregation=self.editkv_config.block_relevance_agg,
        )

        self.editkv_manager.register_prefill(
            request_id=request_id,
            input_ids=input_ids,
            block_attention_scores=block_scores,
            block_table=block_table.tolist(),
            seq_len=seq_len,
        )

        self.editkv_capture.clear()

    def _editkv_on_request_finish(self, request_id: str) -> None:
        """Clean up EditKV state when a request finishes."""
        if self.editkv_manager is not None:
            self.editkv_manager.remove_request(request_id)
