# Copyright (c) Alibaba, Inc. and its affiliates.
# Qwen3-VL + real VideoNSA decoder attention.
#
# This file does NOT modify the original Qwen3-VL implementation.
# It defines a new Qwen3-VL class and only replaces the language decoder
# self-attention modules.
#
# Logic follows the Qwen2.5 VideoNSA / FLA NativeSparseAttention style:
#   q/k/v projection -> RoPE -> g_proj -> parallel_nsa -> o_proj

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from fla.ops.nsa.parallel import parallel_nsa


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _original_attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float,
    scaling: float,
    **kwargs,
):
    """
    Preserve the original Qwen3-VL attention backend.

    This is used as fallback when:
      - VIDEONSA_ENABLE=0
      - generation/KV-cache path q_len != k_len
      - NSA throws shape/kernel errors during early debugging
    """
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        module.config._attn_implementation,
        eager_attention_forward,
    )

    return attention_interface(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        **kwargs,
    )


def real_videonsa_attention_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
):
    """
    Real NSA forward following Qwen2.5 / NativeSparseAttention logic.

    Input Qwen3 shapes:
        hidden_states: [B, L, C]
        query_states: [B, H, L, D]
        key_states:   [B, H_kv, L, D]
        value_states: [B, H_kv, L, D]

    NSA expected shapes:
        q:     [B, L, H, D]
        k:     [B, L, H_kv, D]
        v:     [B, L, H_kv, D]
        g_*:   [B, L, H]
    """
    batch_size, q_len, _ = hidden_states.shape
    k_len = key_states.shape[2]

    # Training/full-prefill path only.
    # For generation with KV cache, q_len may be 1 and k_len long.
    # Keep original dense backend there first; we can optimize generation later.
    if q_len != k_len:
        return None

    # NSA kernels are causal by design. Arbitrary 4D additive masks from HF
    # are not passed into parallel_nsa, matching NativeSparseAttention style.
    # For now we only use NSA on normal full training sequences.
    if attention_mask is not None and attention_mask.ndim not in (2, 4):
        return None

    # Qwen3 gives [B, H, L, D], NSA wants [B, L, H, D].
    q = query_states.transpose(1, 2).contiguous()
    k = key_states.transpose(1, 2).contiguous()
    v = value_states.transpose(1, 2).contiguous()

    # Qwen2.5 NSA logic:
    #   g = g_proj(hidden_states)
    #   g_cmp, g_slc, g_swa = sigmoid(g).unbind(-1)
    g = module.g_proj(hidden_states)
    g = g.view(batch_size, q_len, module.num_heads_for_nsa, 3)
    g_cmp, g_slc, g_swa = g.sigmoid().unbind(-1)

    block_size = _env_int("VIDEONSA_BLOCK_SIZE", 64)
    block_counts = _env_int("VIDEONSA_BLOCK_COUNTS", 16)
    window_size = _env_int("VIDEONSA_WINDOW_SIZE", 512)

    if _env_flag("VIDEONSA_DEBUG_SHAPES", "0") and not getattr(module, "_videonsa_debug_printed", False):
        print(
            "[Qwen3-VideoNSA][real_nsa] "
            f"layer={module.layer_idx}, "
            f"hidden={tuple(hidden_states.shape)}, "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}, "
            f"g_cmp={tuple(g_cmp.shape)}, "
            f"block_size={block_size}, block_counts={block_counts}, window_size={window_size}"
        )
        module._videonsa_debug_printed = True

    o = parallel_nsa(
        q=q,
        k=k,
        v=v,
        g_cmp=g_cmp,
        g_slc=g_slc,
        g_swa=g_swa,
        block_size=block_size,
        block_counts=block_counts,
        window_size=window_size,
    )

    # parallel_nsa returns [B, L, H, D].
    return o, None


class Qwen3VLTextVideoNSAAttention(Qwen3VLTextAttention):
    """
    Drop-in replacement for Qwen3VLTextAttention, but with real NSA.

    It keeps pretrained Qwen3-VL parameters:
      q_proj, k_proj, v_proj, o_proj, q_norm, k_norm

    It adds new trainable parameters:
      g_proj

    Because g_proj is new, loading Qwen3-VL checkpoint needs --strict false.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.num_heads_for_nsa = getattr(
            self,
            "num_attention_heads",
            getattr(self, "num_heads", getattr(config, "num_attention_heads")),
        )

        self.g_proj = nn.Linear(
            config.hidden_size,
            self.num_heads_for_nsa * 3,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        dropout = 0.0 if not self.training else self.attention_dropout

        use_videonsa = _env_flag("VIDEONSA_ENABLE", "1")

        attn_output = None
        attn_weights = None

        if use_videonsa:
            try:
                result = real_videonsa_attention_forward(
                    module=self,
                    hidden_states=hidden_states,
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    attention_mask=attention_mask,
                )
                if result is not None:
                    attn_output, attn_weights = result
            except Exception as e:
                if _env_flag("VIDEONSA_FALLBACK_ON_ERROR", "1"):
                    if not getattr(self, "_videonsa_error_printed", False):
                        print(
                            "[Qwen3-VideoNSA][warning] real NSA failed once; "
                            f"fallback to original attention. error={repr(e)}"
                        )
                        self._videonsa_error_printed = True
                    attn_output = None
                    attn_weights = None
                else:
                    raise

        # Fallback keeps original backend, including flash_attn when --attn_impl flash_attn is used.
        if attn_output is None:
            attn_output, attn_weights = _original_attention_forward(
                module=self,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                dropout=dropout,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3VLForConditionalGenerationVideoNSA(Qwen3VLForConditionalGeneration):
    """
    Qwen3-VL with decoder-side real VideoNSA attention.

    It keeps:
      - Qwen3-VL processor
      - Qwen3-VL vision tower
      - Qwen3-VL patch embedding
      - Qwen3-VL MRoPE

    It replaces only:
      - model.language_model.layers[i].self_attn
    """

    def __init__(self, config):
        super().__init__(config)

        text_config = config.text_config
        layers = self.model.language_model.layers

        for layer_idx, layer in enumerate(layers):
            layer.self_attn = Qwen3VLTextVideoNSAAttention(text_config, layer_idx)

        self._videonsa_video_token_id = int(getattr(config, "video_token_id", 151656))
        self._videonsa_num_layers = len(layers)

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids", None)

        if input_ids is None and len(args) > 0 and torch.is_tensor(args[0]):
            input_ids = args[0]

        video_token_mask = None
        if input_ids is not None and torch.is_tensor(input_ids):
            video_token_mask = input_ids.eq(self._videonsa_video_token_id)

        if _env_flag("VIDEONSA_PRINT_ONCE", "0") and not getattr(self, "_videonsa_printed", False):
            num_video_tokens = int(video_token_mask.sum().item()) if video_token_mask is not None else -1
            print(
                "[Qwen3-VideoNSA] "
                f"decoder_layers={self._videonsa_num_layers}, "
                f"video_token_id={self._videonsa_video_token_id}, "
                f"num_video_tokens={num_video_tokens}, "
                f"enable={os.getenv('VIDEONSA_ENABLE', '1')}, "
                f"real_nsa=parallel_nsa, "
                f"block_size={os.getenv('VIDEONSA_BLOCK_SIZE', '64')}, "
                f"block_counts={os.getenv('VIDEONSA_BLOCK_COUNTS', '16')}, "
                f"window_size={os.getenv('VIDEONSA_WINDOW_SIZE', '512')}"
            )
            self._videonsa_printed = True

        return super().forward(*args, **kwargs)
