import ast
import hashlib
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, SkipValidation, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

import vllm.envs as envs
from vllm.config.parallel import ParallelConfig
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    import vllm.model_executor.layers.quantization as me_quant
    from vllm.config import ModelConfig
else:
    PretrainedConfig = Any
    ModelConfig = Any

    me_quant = LazyLoader(
        "model_executor", globals(), "vllm.model_executor.layers.quantization"
    )

logger = init_logger(__name__)


@config
@dataclass
class EEConfig:
    """Configuration for Early Exit."""

    ramp_config: dict[int, float] | None = None
    """For each ramp, set its layer index and exit threshold."""

    def compute_hash(self) -> str:
        """Compute a hash for the EEConfig."""
        factors = []
        if self.ramp_config:
            factors.append(str(sorted(self.ramp_config.items())))
        else:
            factors.append("None")
        
        hash_str = hashlib.md5(
            str(factors).encode(), usedforsecurity=False
        ).hexdigest()[:10]
        return hash_str