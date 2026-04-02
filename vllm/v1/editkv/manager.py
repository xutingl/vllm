"""EditKV lifecycle manager for vLLM.

Manages the EditKV pipeline:
1. Attention score capture during prefill (GPU or CPU offload)
2. Edit detection and block selection (via RelevanceDict or BlockAttentionMap)
3. Selective recompute scheduling
"""

from __future__ import annotations

import gc
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional

import torch

from vllm.v1.editkv.config import EditKVConfig

# Lazy import to avoid circular dependency — RelevanceDict lives in editkv package
_RelevanceDict = None
_CPURelevanceComputer = None


def _get_relevance_dict_class():
    global _RelevanceDict
    if _RelevanceDict is None:
        from editkv.relevance_dict import RelevanceDict
        _RelevanceDict = RelevanceDict
    return _RelevanceDict


def _get_cpu_computer_class():
    global _CPURelevanceComputer
    if _CPURelevanceComputer is None:
        from editkv.cpu_offload import CPURelevanceComputer
        _CPURelevanceComputer = CPURelevanceComputer
    return _CPURelevanceComputer


class EditKVManager:
    """Manages EditKV state for a vLLM engine.

    Stores per-request relevance dicts (CPU offload) or block attention maps (GPU)
    and handles edit-triggered selective recomputation.
    """

    def __init__(self, config: EditKVConfig):
        self.config = config
        self._request_states: dict[str, EditKVRequestState] = {}
        # Thread pool for CPU relevance computation (shared across requests)
        self._cpu_pool: ThreadPoolExecutor | None = None
        self._cpu_computer: Any | None = None
        if config.cpu_offload_enabled:
            self._cpu_pool = ThreadPoolExecutor(max_workers=1)

    def init_cpu_computer(self, model: torch.nn.Module) -> None:
        """Initialize the CPU relevance computer from model weights (call once)."""
        if not self.config.cpu_offload_enabled:
            return
        ComputerCls = _get_cpu_computer_class()
        self._cpu_computer = ComputerCls.from_model(
            model,
            threshold=self.config.cpu_relevance_threshold,
            fallback_ratio=self.config.cpu_fallback_ratio,
            num_threads=self.config.cpu_num_threads,
        )

    def register_prefill(
        self,
        request_id: str,
        input_ids: list[int],
        block_attention_scores: Any,
        block_table: list[int],
        seq_len: int,
    ) -> None:
        """Register a completed prefill with captured attention scores."""
        self._request_states[request_id] = EditKVRequestState(
            request_id=request_id,
            original_input_ids=input_ids,
            block_attention_scores=block_attention_scores,
            block_table=block_table,
            seq_len=seq_len,
        )

    def register_prefill_cpu_offload(
        self,
        request_id: str,
        input_ids: list[int],
        cpu_h: list[torch.Tensor | None],
        cpu_k: list[torch.Tensor | None],
        block_table: list[int],
        seq_len: int,
    ) -> None:
        """Register prefill and submit background CPU relevance computation."""
        state = EditKVRequestState(
            request_id=request_id,
            original_input_ids=input_ids,
            block_attention_scores=None,
            block_table=block_table,
            seq_len=seq_len,
        )
        self._request_states[request_id] = state

        if self._cpu_pool is not None and self._cpu_computer is not None:
            future = self._cpu_pool.submit(self._cpu_computer.compute, cpu_h, cpu_k, seq_len)
            state.cpu_compute_future = future

    def get_request_state(self, request_id: str) -> Optional["EditKVRequestState"]:
        return self._request_states.get(request_id)

    def compute_blocks_to_recompute(
        self,
        request_id: str,
        edited_input_ids: list[int],
        block_size: int = 16,
    ) -> tuple[list[int], bool]:
        """Determine which blocks need recomputation for an edited request.

        Supports both CPU offload (RelevanceDict) and GPU (BlockAttentionMap) paths.

        Args:
            request_id: ID of the original (pre-edit) request.
            edited_input_ids: Token IDs of the edited document.
            block_size: Tokens per block.

        Returns:
            Tuple of (selected_block_indices, full_recompute_triggered).
        """
        state = self._request_states.get(request_id)
        if state is None:
            num_blocks = (len(edited_input_ids) + block_size - 1) // block_size
            return list(range(num_blocks)), True

        # Find edited token positions
        edited_positions = _find_edited_positions(
            state.original_input_ids, edited_input_ids
        )

        if not edited_positions:
            return [], False

        # Map edited positions to blocks
        edit_blocks = sorted(set(pos // block_size for pos in edited_positions))

        # --- CPU offload path: use RelevanceDict ---
        if state.cpu_compute_future is not None:
            # Wait for CPU computation if not done
            try:
                relevance_dict = state.cpu_compute_future.result(timeout=60)
                state.relevance_dict = relevance_dict
            except Exception:
                # Timeout or error — fall back to full recompute
                num_blocks = (max(len(edited_input_ids), state.seq_len) + block_size - 1) // block_size
                return list(range(num_blocks)), True

        if state.relevance_dict is not None:
            affected = state.relevance_dict.lookup_multi(edit_blocks)
            if affected is None:
                num_blocks = (max(len(edited_input_ids), state.seq_len) + block_size - 1) // block_size
                return list(range(num_blocks)), True
            all_selected = sorted(set(edit_blocks) | set(affected))
            return all_selected, False

        # --- GPU path: use BlockAttentionMap (legacy) ---
        if len(edited_input_ids) != state.seq_len:
            num_blocks = (len(edited_input_ids) + block_size - 1) // block_size
            return list(range(num_blocks)), True

        if state.block_attention_scores is None:
            min_edit_block = min(edit_blocks)
            num_blocks = (state.seq_len + block_size - 1) // block_size
            return list(range(min_edit_block, num_blocks)), False

        edit_block_tensor = torch.tensor(edit_blocks, dtype=torch.long, device="cuda")
        affected = state.block_attention_scores.get_affected_blocks(edit_block_tensor)
        affected_list = affected.tolist()

        min_edit_block = min(edit_blocks)
        num_blocks = (state.seq_len + block_size - 1) // block_size
        candidate_blocks = [
            b for b in range(min_edit_block + 1, num_blocks)
            if b not in edit_blocks
        ]

        num_to_select = max(1, int(self.config.recompute_ratio * len(candidate_blocks)))
        num_to_select = min(num_to_select, len(candidate_blocks))

        affected_set = set(affected_list)
        selected = sorted(
            candidate_blocks,
            key=lambda b: (b not in affected_set, b),
        )[:num_to_select]

        all_selected = sorted(set(edit_blocks) | set(selected))
        return all_selected, False

    def remove_request(self, request_id: str) -> None:
        """Clean up state for a completed request."""
        state = self._request_states.pop(request_id, None)
        if state is not None:
            del state
            gc.collect()

    def shutdown(self) -> None:
        """Shut down the CPU thread pool."""
        if self._cpu_pool is not None:
            self._cpu_pool.shutdown(wait=False)
            self._cpu_pool = None


class EditKVRequestState:
    """Per-request state for EditKV."""

    def __init__(
        self,
        request_id: str,
        original_input_ids: list[int],
        block_attention_scores: Any,
        block_table: list[int],
        seq_len: int,
    ):
        self.request_id = request_id
        self.original_input_ids = original_input_ids
        self.block_attention_scores = block_attention_scores
        self.block_table = block_table
        self.seq_len = seq_len
        self.created_at = time.time()
        # CPU offload state
        self.relevance_dict: Any | None = None
        self.cpu_compute_future: Future | None = None


def _find_edited_positions(
    original_ids: list[int],
    edited_ids: list[int],
) -> list[int]:
    """Find token positions that differ between original and edited sequences."""
    max_len = max(len(original_ids), len(edited_ids))
    positions = []
    for i in range(max_len):
        orig = original_ids[i] if i < len(original_ids) else -1
        edit = edited_ids[i] if i < len(edited_ids) else -1
        if orig != edit:
            positions.append(i)
    return positions
