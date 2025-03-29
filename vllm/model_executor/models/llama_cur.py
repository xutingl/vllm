# modified from https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/llama/modeling_llama.py
# and adapts the parallel decoding from https://github.com/raymin0223/fast_robust_early_exit
# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
import warnings
from typing import List, Optional, Tuple, Union, Iterable, Set, Dict, Any, Type

import torch
import torch.distributed as dist
import torch.utils.checkpoint
from .llama_ee_utils import BetaMixture1D, get_skip_mask
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import (
    GreedySearchDecoderOnlyOutput,
    GreedySearchEncoderDecoderOutput,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.utils import logging

from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

GreedySearchOutput = Union[
    GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput
]

logger = logging.get_logger(__name__)



class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x

class LlamaAttention(nn.Module):

    def __init__(self,
                 config: LlamaConfig,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 rope_theta: float = 10000,
                 rope_scaling: Optional[Dict[str, Any]] = None,
                 max_position_embeddings: int = 8192,
                 quant_config: Optional[QuantizationConfig] = None,
                 bias: bool = False,
                 bias_o_proj: bool = False,
                 cache_config: Optional[CacheConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        # Phi models introduced a partial_rotary_factor parameter in the config
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )

        if hasattr(config, "interleaved_sliding_window"):
            interleaved_sliding_window = config.interleaved_sliding_window
            if isinstance(interleaved_sliding_window, int):
                sliding_window = interleaved_sliding_window
            elif isinstance(interleaved_sliding_window, list):
                sw_idx = layer_idx % len(interleaved_sliding_window)
                sliding_window = interleaved_sliding_window[sw_idx]
            else:
                raise ValueError(
                    f"{type(interleaved_sliding_window)} is not supported.")
        else:
            sliding_window = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        skip_mask=False,
        stack_hidden_states=None,
    ) -> torch.Tensor:
        
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(
            self,
            config: LlamaConfig,
            cache_config: Optional[CacheConfig] = None,
            prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = LlamaAttention(config=config,
                                        hidden_size=config.hidden_size,
                                        num_heads=config.num_attention_heads,
                                        num_kv_heads=getattr(config, "num_key_value_heads",
                                                             config.num_attention_heads),
                                                             rope_theta=rope_theta,
                                        rope_scaling=rope_scaling,
                                        max_position_embeddings=max_position_embeddings,
                                        cache_config=cache_config,
                                        prefix=f"{prefix}.self_attn")
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        skip_mask=False,
        stack_hidden_states=None,
        parallel_mask=False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `kv_caches` key value states are returned and can be used to speed up decoding
                (see `kv_caches`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states

            vLLM modifications:
            past_key_value: Optional[Tuple[torch.Tensor]] = None     is changed to
            kv_cache: torch.Tensor,
        """

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(
            positions,
            hidden_states,
            kv_cache,
            attn_metadata,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            use_cache=use_cache,
            skip_mask=skip_mask,
            stack_hidden_states=stack_hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

class LlamaModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: Type[LlamaDecoderLayer] = LlamaDecoderLayer,):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        # self.embed_tokens = nn.Embedding(
        #     config.vocab_size, config.hidden_size, self.padding_idx
        # )
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, cache_config=cache_config, prefix=f"{prefix}.layers.{i}") for i in range(config.num_hidden_layers)]
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.gradient_checkpointing = False
        

        # Shallow-Deep Module
        self.use_shallow_deep = config.use_shallow_deep
        self.shallow_exit_layer = config.shallow_exit_layer

        # Synchronized Parallel Decoding
        self.block_op = [
            0
        ] * config.num_hidden_layers  # to calculate the average number of forward block layers
        self.parallel_tokens_shallow = (
            0  # how much tokens are used in parallel decoding as stack_hidden_states
        )
        self.parallel_tokens_deep = (
            0  # how much tokens are used in parallel decoding with skip_mask = False
        )
        self.stack_hidden_states = ()  # store hidden_states that do not forward Deep decoder

        # Adaptive Threshold Estimator
        #self.bmm_model = BetaMixture1D()
        self.bmm_threshold = config.shallow2deep_conf_threshold
        self.stack_conf, self.stack_pred = (), ()
        self.stack_conf_all, self.stack_ident_all = (), []
        self.exited_rates = [0, 0]

        self.block_op = [
            0
        ] * config.num_hidden_layers  # to calculate the average number of forward block layers

        self.config.optimal = False # Disable optimal for vLLM
        if self.config.optimal:
            self.optimal_exiting_layers = []

    # def get_input_embeddings(self):
    #     return self.embed_tokens
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, kv_caches_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                kv_caches_length=kv_caches_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        lm_head: Optional[nn.Linear] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # print(f"LlamaModel forward. input_ids shape: {input_ids.shape}")
        # print(f"attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
        # print("-------------------------")
        if get_pp_group().is_first_rank:
            #print(f"Is first rank. input ids shape: {input_ids.shape}")
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            #print(f"Not first rank. input ids shape: {input_ids.shape}")
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        
        print(f"hidden_states shape: {hidden_states.shape}")

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            # batch_size, seq_length = input_ids.shape

            # Change to adapt vLLM
            batch_size = 1
            seq_length = input_ids.shape[0]

        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        kv_caches_length = 0

        # if kv_caches is not None:
        #     kv_caches = None
        #     print(f"Force kv_caches to None")
            # if len(kv_caches[0]) > 0:
            #     print(f"len(kv_caches): {len(kv_caches)}")
            #     print(f"kv_caches[-1]: {kv_caches[-1]}")
            #     kv_caches_length = kv_caches[-1][0].shape[2]
            #     seq_length_with_past = seq_length_with_past + kv_caches_length
            # else:
            #     print(f"kv_caches is not none but is empty: {kv_caches}")

        if positions is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            positions = torch.arange(
                kv_caches_length,
                seq_length + kv_caches_length,
                dtype=torch.long,
                device=device,
            )
            positions = positions.unsqueeze(0).view(-1, seq_length)
        else:
            positions = positions.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        # WARNING
        attention_mask = None
        # attention_mask = self._prepare_decoder_attention_mask(
        #     attention_mask,
        #     (batch_size, seq_length),
        #     inputs_embeds,
        #     kv_caches_length,
        # )


        ## TODO FIX THIS
        # initialize kv_caches with `None` if past does not exist
        if kv_caches is None:
            self.stack_hidden_states = ()
            self.stack_conf, self.stack_pred = (), ()
            # self.exited_rates = [0, 0]
        skip_mask = False  # False: forward, and True: skip
        self.shallow2deep = False  # False: skip, and True: forward
        self.lm_logits = None  # to prevent calculating logits twice

        # hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        if self.config.optimal:
            exit_decision = [None] * self.config.num_hidden_layers

        for idx, decoder_layer in enumerate(self.layers):

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                kv_caches[idx] if kv_caches is not None else None
            )

            # auto_reg = True if hidden_states.shape[1] == 1 else False
            auto_reg = True

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    positions,
                    None,
                )
            else:
                # shallow-deep module
                if auto_reg and idx > 0:
                    # self.use_shallow_deep = False # [Turn off EE]
                    if self.use_shallow_deep and idx == self.shallow_exit_layer:
                        lm_logits = lm_head(self.norm(hidden_states))
                        print(f"logits shape: {lm_logits.shape}")
                        skip_mask, conf = get_skip_mask(
                            logits=lm_logits,
                            hidden_states=hidden_states,
                            ee_policy="eager",
                            config=self.config,
                            adapt_threshold=self.bmm_threshold,
                            return_conf=True,
                        )
                        print(f"skip_mask: {skip_mask}, conf: {conf}")
                        self.stack_conf = self.stack_conf + (conf,)
                        self.stack_pred = self.stack_pred + (lm_logits,)
                        if skip_mask:
                            self.exited_rates[0] += 1
                        else:
                            self.exited_rates[1] += 1
                        # print(f"--------[LlamaModel: forward] EE statistics at layer{idx}---------")
                        # print(f"self.exited_rates([num_ee, num_no_ee]): {self.exited_rates}")
                        # print(f"skip_mask: {skip_mask}, conf: {conf}")
                        # print(f"--------[LlamaModel: forward] EE statistics---------")

                        if skip_mask:
                            # print(f"[LlamaModel: forward] SKIPPING!")
                            self.lm_logits = lm_logits
                            if self.config.parallel_gen_token:
                                if use_cache:
                                    for j in range(idx, len(self.layers)):
                                        next_decoder_cache = next_decoder_cache + (
                                            kv_caches[j],
                                        )
                                self.stack_hidden_states = self.stack_hidden_states + (
                                    hidden_states,
                                )
                            break
                        else:
                            self.shallow2deep = True
                            if self.config.parallel_gen_token and len(
                                self.stack_hidden_states
                            ):
                                self.parallel_tokens_shallow += len(
                                    self.stack_hidden_states
                                )
                                self.parallel_tokens_deep += 1

                                # in Shallow-Deep decoder, generate the next token in a non-autoregressive manner
                                hidden_states, next_decoder_cache = (
                                    self.parallel_gen_token(
                                        hidden_states=hidden_states,
                                        attention_mask=attention_mask,
                                        positions=positions,
                                        kv_caches=kv_caches,
                                        output_attentions=False,
                                        use_cache=use_cache,
                                        skip_mask=skip_mask,
                                        stack_hidden_states=self.stack_hidden_states,
                                        layer_idx=self.shallow_exit_layer,
                                        next_decoder_cache=next_decoder_cache,
                                    )
                                )

                                # Adaptive Threshold Estimator
                                if self.config.use_adapt_threshold:
                                    # Calibration Set Update
                                    self.lm_logits = lm_head(self.norm(hidden_states))
                                    deep_pred = self.lm_logits.argmax(-1)
                                    shallow_pred = (
                                        torch.cat(self.stack_pred).argmax(-1).view(-1)
                                    )

                                    self.stack_conf_all += self.stack_conf
                                    self.stack_ident_all += list(
                                        (deep_pred.view(-1) == shallow_pred.view(-1))
                                        .long()
                                        .cpu()
                                        .numpy(),
                                    )
                                    self.stack_conf, self.stack_pred = (), ()
                                break

                hidden_states, residual = decoder_layer(
                    hidden_states,
                    kv_caches[idx],
                    attn_metadata,
                    residual,
                    attention_mask=attention_mask,
                    positions=positions,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                # if auto_reg and self.config.optimal:
                #     lm_logits = lm_head(self.norm(layer_outputs[0]))
                #     exit_decision[idx] = torch.argmax(lm_logits, dim=-1)
                    # print(lm_logits.shape)
                    # print(torch.argmax(lm_logits, dim=-1))

            # ??? why need this?
            # hidden_states = layer_outputs[0]

            # if use_cache:
            #     next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            # if output_attentions:
            #     all_self_attns += (layer_outputs[1],)

        hidden_states, _ = self.norm(hidden_states, residual)

        # print(f"[LlamaModel: forward] Returning hidden_states shape: {hidden_states.shape}")
        return hidden_states
        if self.config.optimal and auto_reg:
            self.lm_logits = lm_head(hidden_states)
            final_prediction = torch.argmax(self.lm_logits, dim=-1)
            for layer_idx, prediction in enumerate(exit_decision):
                if torch.all(prediction == final_prediction):
                    self.optimal_exiting_layers.append(layer_idx)
                    break
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        
        return hidden_states
        # return BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        # )

    def parallel_gen_token(
        self,
        hidden_states=None,
        attention_mask=None,
        positions=None,
        kv_caches=None,
        output_attentions=False,
        use_cache=False,
        skip_mask=None,
        stack_hidden_states=None,
        layer_idx=None,
        next_decoder_cache=None,
    ):
        if not self.config.copy_skipped_hidden_states:
            hidden_states = torch.cat(
                self.stack_hidden_states + (hidden_states,), dim=1
            )
            extended_attention_mask, positions = None, None
        else:
            self.stack_hidden_states = torch.cat(self.stack_hidden_states, dim=1)

        if extended_attention_mask is None:
            batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
            seq_length_with_past = seq_length
            kv_caches_length = 0

            if kv_caches is not None:
                kv_caches_length = kv_caches[-1][0].shape[2]
                seq_length_with_past = seq_length_with_past + kv_caches_length

            if positions is None:
                device = hidden_states.device
                positions = torch.arange(
                    kv_caches_length,
                    seq_length + kv_caches_length,
                    dtype=torch.long,
                    device=device,
                )
                positions = positions.unsqueeze(0).view(-1, seq_length)
            else:
                positions = positions.view(-1, seq_length).long()

            # embed positions
            if extended_attention_mask is None:
                extended_attention_mask = torch.ones(
                    (batch_size, seq_length_with_past),
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
            extended_attention_mask = self._prepare_decoder_attention_mask(
                extended_attention_mask,
                (batch_size, seq_length),
                hidden_states,
                kv_caches_length,
            )

        # print(f"parallel decode: len(kv_caches) {len(kv_caches)}")
        for j in range(layer_idx, len(self.layers)):
            # print(f"parallel decode: layer {j}")
            # print(f"parallel decode: len(kv_caches) {len(kv_caches)}")
            assert (
                kv_caches is not None
            ), "past_key_value is required for parallel decoding"

            layer_outputs = self.layers[j](
                hidden_states,
                attention_mask=extended_attention_mask,
                positions=positions,
                past_key_value=kv_caches[j],
                output_attentions=output_attentions,
                use_cache=use_cache,
                skip_mask=False,
                parallel_mask=True,
                stack_hidden_states=(
                    self.stack_hidden_states
                    if self.config.copy_skipped_hidden_states
                    else None
                ),
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        self.stack_hidden_states = ()

        return hidden_states, next_decoder_cache
    
    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class LlamaForCausalLM(nn.Module, VllmModelForTextGeneration):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    embedding_padding_modules = ["lm_head"]

    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm"
    }
    def __init__(self, *, vllm_config: VllmConfig, apparate=True, optimal=False, prefix: str = "",):
        config = vllm_config.model_config.hf_config
        
        super().__init__()
        self.unpadded_vocab_size = config.vocab_size
        self.apparate_config = {
            "use_shallow_deep": True,
            "shallow_exit_layer": 20,
            "shallow2deep_conf_type": "softmax",
            "shallow2deep_conf_threshold": 0.6,
            "parallel_gen_token": False,
            "rollback_conf_threshold": None,
            "parallel_causal_mask": True,
            "copy_skipped_hidden_states": False,
            "use_adapt_threshold": False,
            "apparate": apparate,
            "optimal": optimal,
        }
        self.times = []
        config.update(self.apparate_config)
        config.output_hidden_states = True # Output hidden states

        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config

        self.model = LlamaModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config else
                    lora_config.lora_vocab_padding_size),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()
        self.sampler = get_sampler()
        

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)



    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""
        use_cache = False # Disable the cache in llama code becuase we are using vLLM's kv cache

        # if input_ids.dim() == 1:
        #     print(f"input_ids shape : {input_ids.shape}")
    

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            lm_head=self.compute_logits,
            intermediate_tensors=intermediate_tensors,
        )

        if self.model.shallow2deep:
            self.model.stack_conf, self.model.stack_pred = (), ()
        # print(f"[LlamaForCausalLM forward] Model output shape: {outputs.shape}")
        return outputs

        # hidden_states = outputs[0]
        hidden_states = outputs[0]
        if self.model.lm_logits is None:
            # logits = self.lm_head(hidden_states)
            logits = self.compute_logits(hidden_states, None)
        else:
            logits = self.model.lm_logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # def prepare_inputs_for_generation(
    #     self,
    #     input_ids,
    #     kv_caches=None,
    #     attention_mask=None,
    #     inputs_embeds=None,
    #     **kwargs,
    # ):
    #     if kv_caches:
    #         input_ids = input_ids[:, -1:]

    #     positions = kwargs.get("positions", None)
    #     if attention_mask is not None and positions is None:
    #         # create positions on the fly for batch generation
    #         positions = attention_mask.long().cumsum(-1) - 1
    #         positions.masked_fill_(attention_mask == 0, 1)
    #         if kv_caches:
    #             positions = positions[:, -1].unsqueeze(-1)

    #     # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    #     if inputs_embeds is not None and kv_caches is None:
    #         model_inputs = {"inputs_embeds": inputs_embeds}
    #     else:
    #         model_inputs = {"input_ids": input_ids}

    #     model_inputs.update(
    #         {
    #             "positions": positions,
    #             "kv_caches": kv_caches,
    #             "use_cache": kwargs.get("use_cache"),
    #             "attention_mask": attention_mask,
    #         }
    #     )
    #     return model_inputs

    # @staticmethod
    # def _reorder_cache(kv_caches, beam_idx):
    #     reordered_past = ()
    #     for layer_past in kv_caches:
    #         reordered_past += (
    #             tuple(
    #                 past_state.index_select(0, beam_idx) for past_state in layer_past
    #             ),
    #         )
    #     return reordered_past

    # def greedy_search(
    #     self,
    #     input_ids: torch.LongTensor,
    #     logits_processor: Optional[LogitsProcessorList] = None,
    #     stopping_criteria: Optional[StoppingCriteriaList] = None,
    #     max_length: Optional[int] = None,
    #     pad_token_id: Optional[int] = None,
    #     eos_token_id: Optional[Union[int, List[int]]] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     output_scores: Optional[bool] = None,
    #     return_dict_in_generate: Optional[bool] = None,
    #     synced_gpus: bool = False,
    #     streamer: Optional["BaseStreamer"] = None,
    #     **model_kwargs,
    # ) -> Union[GreedySearchOutput, torch.LongTensor]:
    #     r"""
    #     Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
    #     used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    #     <Tip warning={true}>

    #     In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
    #     instead. For an overview of generation strategies and code examples, check the [following
    #     guide](../generation_strategies).
    #     """

    #     # init values
    #     logits_processor = (
    #         logits_processor if logits_processor is not None else LogitsProcessorList()
    #     )
    #     stopping_criteria = (
    #         stopping_criteria
    #         if stopping_criteria is not None
    #         else StoppingCriteriaList()
    #     )
    #     if max_length is not None:
    #         warnings.warn(
    #             "`max_length` is deprecated in this function, use"
    #             " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
    #             UserWarning,
    #         )
    #         stopping_criteria = validate_stopping_criteria(
    #             stopping_criteria, max_length
    #         )
    #     pad_token_id = (
    #         pad_token_id
    #         if pad_token_id is not None
    #         else self.generation_config.pad_token_id
    #     )
    #     eos_token_id = (
    #         eos_token_id
    #         if eos_token_id is not None
    #         else self.generation_config.eos_token_id
    #     )
    #     if isinstance(eos_token_id, int):
    #         eos_token_id = [eos_token_id]
    #     eos_token_id_tensor = (
    #         torch.tensor(eos_token_id).to(input_ids.device)
    #         if eos_token_id is not None
    #         else None
    #     )
    #     output_scores = (
    #         output_scores
    #         if output_scores is not None
    #         else self.generation_config.output_scores
    #     )
    #     output_attentions = (
    #         output_attentions
    #         if output_attentions is not None
    #         else self.generation_config.output_attentions
    #     )
    #     output_hidden_states = (
    #         output_hidden_states
    #         if output_hidden_states is not None
    #         else self.generation_config.output_hidden_states
    #     )
    #     return_dict_in_generate = (
    #         return_dict_in_generate
    #         if return_dict_in_generate is not None
    #         else self.generation_config.return_dict_in_generate
    #     )

    #     # init attention / hidden states / scores tuples
    #     scores = () if (return_dict_in_generate and output_scores) else None
    #     decoder_attentions = (
    #         () if (return_dict_in_generate and output_attentions) else None
    #     )
    #     cross_attentions = (
    #         () if (return_dict_in_generate and output_attentions) else None
    #     )
    #     decoder_hidden_states = (
    #         () if (return_dict_in_generate and output_hidden_states) else None
    #     )

    #     # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    #     if return_dict_in_generate and self.config.is_encoder_decoder:
    #         encoder_attentions = (
    #             model_kwargs["encoder_outputs"].get("attentions")
    #             if output_attentions
    #             else None
    #         )
    #         encoder_hidden_states = (
    #             model_kwargs["encoder_outputs"].get("hidden_states")
    #             if output_hidden_states
    #             else None
    #         )

    #     # keep track of which sequences are already finished
    #     unfinished_sequences = torch.ones(
    #         input_ids.shape[0], dtype=torch.long, device=input_ids.device
    #     )

    #     this_peer_finished = False  # used by synced_gpus only

    #     idx = 0
    #     while True:
    #         if synced_gpus:
    #             # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
    #             # The following logic allows an early break if all peers finished generating their sequence
    #             this_peer_finished_flag = torch.tensor(
    #                 0.0 if this_peer_finished else 1.0
    #             ).to(input_ids.device)
    #             # send 0.0 if we finished, 1.0 otherwise
    #             dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
    #             # did all peers finish? the reduced sum will be 0.0 then
    #             if this_peer_finished_flag.item() == 0.0:
    #                 break

    #         # prepare model inputs
    #         model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

    #         start = torch.cuda.Event(enable_timing=True)
    #         end = torch.cuda.Event(enable_timing=True)
    #         start.record()
    #         # forward pass to get next token
    #         outputs = self(
    #             **model_inputs,
    #             return_dict=True,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #         )
    #         idx += 1
    #         end.record()
    #         torch.cuda.synchronize()

    #         if idx > 1:
    #             self.times.append(start.elapsed_time(end))

    #         if synced_gpus and this_peer_finished:
    #             continue  # don't waste resources running the code we don't need

    #         next_token_logits = outputs.logits[:, -1, :]

    #         # pre-process distribution
    #         next_tokens_scores = logits_processor(input_ids, next_token_logits)

    #         # Store scores, attentions and hidden_states when required
    #         if return_dict_in_generate:
    #             if output_scores:
    #                 scores += (next_tokens_scores,)
    #             if output_attentions:
    #                 decoder_attentions += (
    #                     (outputs.decoder_attentions,)
    #                     if self.config.is_encoder_decoder
    #                     else (outputs.attentions,)
    #                 )
    #                 if self.config.is_encoder_decoder:
    #                     cross_attentions += (outputs.cross_attentions,)

    #             if output_hidden_states:
    #                 decoder_hidden_states += (
    #                     (outputs.decoder_hidden_states,)
    #                     if self.config.is_encoder_decoder
    #                     else (outputs.hidden_states,)
    #                 )

    #         # argmax
    #         next_tokens = torch.argmax(next_tokens_scores, dim=-1)

    #         # finished sentences should have their next token be a padding token
    #         if eos_token_id is not None:
    #             if pad_token_id is None:
    #                 raise ValueError(
    #                     "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
    #                 )
    #             next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
    #                 1 - unfinished_sequences
    #             )

    #         # update generated ids, model inputs, and length for next step
    #         input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

    #         if streamer is not None:
    #             streamer.put(next_tokens.cpu())
    #         model_kwargs = self._update_model_kwargs_for_generation(
    #             outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
    #         )

    #         # if eos_token was found in one sentence, set sentence to finished
    #         if eos_token_id_tensor is not None:
    #             unfinished_sequences = unfinished_sequences.mul(
    #                 next_tokens.tile(eos_token_id_tensor.shape[0], 1)
    #                 .ne(eos_token_id_tensor.unsqueeze(1))
    #                 .prod(dim=0)
    #             )

    #             # stop when each sentence is finished
    #             if unfinished_sequences.max() == 0:
    #                 this_peer_finished = True

    #         # stop if we exceed the maximum length
    #         if stopping_criteria(input_ids, scores):
    #             this_peer_finished = True

    #         if this_peer_finished and not synced_gpus:
    #             break
            
    #     if streamer is not None:
    #         streamer.end()

    #     if return_dict_in_generate:
    #         if self.config.is_encoder_decoder:
    #             return GreedySearchEncoderDecoderOutput(
    #                 sequences=input_ids,
    #                 scores=scores,
    #                 encoder_attentions=encoder_attentions,
    #                 encoder_hidden_states=encoder_hidden_states,
    #                 decoder_attentions=decoder_attentions,
    #                 cross_attentions=cross_attentions,
    #                 decoder_hidden_states=decoder_hidden_states,
    #             )
    #         else:
    #             return GreedySearchDecoderOnlyOutput(
    #                 sequences=input_ids,
    #                 scores=scores,
    #                 attentions=decoder_attentions,
    #                 hidden_states=decoder_hidden_states,
    #             )
    #     else:
    #         return input_ids
        
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata = None,
    ) -> Optional[torch.Tensor]:
        
        # hidden_states = hidden_states.squeeze()
        # print(f"[compute_logits] hidden_states shape: {hidden_states.shape}. lm_head shape: {self.lm_head}")
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight in weights)

    # This function is used to remap the mistral format as
    # used by Mistral and Llama <=2
    def maybe_remap_mistral(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:

        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.config.head_dim * n_heads
            attn_out = self.config.hidden_size

            return w.view(n_heads, attn_in // n_heads // 2, 2,
                          attn_out).transpose(1, 2).reshape(attn_in, attn_out)

        mapping = self.mistral_mapping
        modules = name.split(".")

        # rotary embeds should be sliced
        if "wk" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_key_value_heads)
        elif "wq" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_attention_heads)

        for item in modules:
            if item in mapping and mapping[item] not in name:
                name = name.replace(item, mapping[item])

        return name, loaded_weight