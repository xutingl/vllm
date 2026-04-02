"""EditKV configuration for vLLM integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class EditKVConfig:
    """Configuration for EditKV selective KV cache recomputation.

    Attributes:
        enabled: Whether EditKV is active.
        storage_backend: Where to store block attention maps ("gpu" or "cpu").
        top_k_blocks: Number of top attending blocks to store per key block.
        recompute_ratio: Fraction of non-edit candidate blocks to recompute.
        cv_threshold: Coefficient of variation threshold for global impact detection.
        block_relevance_agg: How to aggregate token scores within a block.
        capture_attention_scores: Whether to capture attention scores during prefill.
        cpu_offload_enabled: Use CPU offloading for relevance computation (new design).
        cpu_relevance_threshold: Attention score threshold for relevance dict.
        cpu_fallback_ratio: If more than this fraction of suffix affected, full recompute.
        cpu_num_threads: Thread pool size for CPU relevance computation.
    """
    enabled: bool = False
    storage_backend: Literal["gpu", "cpu"] = "gpu"
    top_k_blocks: int = 8
    recompute_ratio: float = 0.2
    cv_threshold: float = 0.01
    block_relevance_agg: Literal["mean", "max"] = "max"
    capture_attention_scores: bool = True
    cpu_offload_enabled: bool = True
    cpu_relevance_threshold: float = 0.01
    cpu_fallback_ratio: float = 0.5
    cpu_num_threads: int = 4
