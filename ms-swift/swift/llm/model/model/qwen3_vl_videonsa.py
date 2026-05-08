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

from nsa.nsa import nsa_func


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


def videonsa_mixed_attention_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float,
    **kwargs,
):
    """
    VideoNSA-style mixed decoder attention.

    Text tokens:
        original Qwen3-VL attention

    Video tokens:
        NSA attention

    Final:
        scatter video NSA output back into the dense output.
    """

    batch_size, q_len, _ = hidden_states.shape
    k_len = key_states.shape[2]

    # Generation decode with KV cache: q_len is usually 1, k_len is long.
    # For now, keep original Qwen3-VL attention.
    if q_len != k_len:
        return _original_attention_forward(
            module=module,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            dropout=dropout,
            scaling=module.scaling,
            **kwargs,
        )

    vision_mask = getattr(module, "_videonsa_vision_mask", None)

    # If no vision mask, fallback to original attention.
    if vision_mask is None:
        return _original_attention_forward(
            module=module,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            dropout=dropout,
            scaling=module.scaling,
            **kwargs,
        )

    vision_mask = vision_mask.to(hidden_states.device)

    # Mask shape must match current full-prefill sequence.
    if vision_mask.shape[0] != batch_size or vision_mask.shape[1] != q_len:
        return _original_attention_forward(
            module=module,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            dropout=dropout,
            scaling=module.scaling,
            **kwargs,
        )

    # 1. Original Qwen3-VL attention for all tokens.
    # Text tokens will keep this output.
    dense_output, dense_weights = _original_attention_forward(
        module=module,
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        attention_mask=attention_mask,
        dropout=dropout,
        scaling=module.scaling,
        **kwargs,
    )
    # dense_output: [B, L, H, D]

    # 2. Gate projection for NSA branches.
    g = module.g_proj(hidden_states)
    g = g.view(batch_size, q_len, module.num_heads_for_nsa, 3)
    g_cmp, g_slc, g_swa = g.sigmoid().unbind(-1)
    # each: [B, L, H]

    block_size = _env_int("VIDEONSA_BLOCK_SIZE", 64)
    block_count = _env_int("VIDEONSA_BLOCK_COUNTS", 16)
    window_size = _env_int("VIDEONSA_WINDOW_SIZE", 512)

    mixed_output = dense_output.clone()

    # 3. Only video tokens go through NSA.
    for b in range(batch_size):
        video_idx = torch.nonzero(vision_mask[b], as_tuple=False).squeeze(-1)

        if video_idx.numel() == 0:
            continue

        # Qwen3 states:
        #   query_states: [B, H, L, D]
        # NSA wants:
        #   q/k/v: [B, L, H, D]
        q_v = query_states[b:b + 1, :, video_idx, :].transpose(1, 2).contiguous()
        k_v = key_states[b:b + 1, :, video_idx, :].transpose(1, 2).contiguous()
        v_v = value_states[b:b + 1, :, video_idx, :].transpose(1, 2).contiguous()

        g_cmp_v = g_cmp[b:b + 1, video_idx, :].contiguous()
        g_slc_v = g_slc[b:b + 1, video_idx, :].contiguous()
        g_swa_v = g_swa[b:b + 1, video_idx, :].contiguous()

        video_len = int(video_idx.numel())

        cur_block_count = min(
            block_count,
            max(1, (video_len + block_size - 1) // block_size),
        )
        cur_window_size = min(window_size, video_len)

        nsa_output = nsa_func(
            q=q_v,
            k=k_v,
            v=v_v,
            g_cmp=g_cmp_v,
            g_slc=g_slc_v,
            g_swa=g_swa_v,
            block_count=cur_block_count,
            block_size=block_size,
            window_size=cur_window_size,
            scale=module.scaling,
            return_attn_weights=False,
            layer_idx=getattr(module, "layer_idx", None),
        )
        # nsa_output: [1, num_video_tokens, H, D]

        mixed_output[b:b + 1, video_idx, :, :] = nsa_output

    return mixed_output, dense_weights


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
                attn_output, attn_weights = videonsa_mixed_attention_forward(
                    module=self,
                    hidden_states=hidden_states,
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    attention_mask=attention_mask,
                    dropout=dropout,
                    **kwargs,
                )
            except Exception as e:
                if _env_flag("VIDEONSA_FALLBACK_ON_ERROR", "1"):
                    if not getattr(self, "_videonsa_error_printed", False):
                        print(
                            "[Qwen3-VideoNSA][warning] mixed VideoNSA failed once; "
                            f"fallback to original attention. error={repr(e)}"
                        )
                        self._videonsa_error_printed = True

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
                else:
                    raise
        else:
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
                f"mode=mixed_text_dense_video_nsa, "
                f"block_size={os.getenv('VIDEONSA_BLOCK_SIZE', '64')}, "
                f"block_counts={os.getenv('VIDEONSA_BLOCK_COUNTS', '16')}, "
                f"window_size={os.getenv('VIDEONSA_WINDOW_SIZE', '512')}"
            )
            self._videonsa_printed = True

        # Pass video mask to every decoder attention layer.
        for layer in self.model.language_model.layers:
            layer.self_attn._videonsa_vision_mask = video_token_mask

        try:
            return super().forward(*args, **kwargs)
        finally:
            # Avoid stale mask being reused in the next forward.
            for layer in self.model.language_model.layers:
                layer.self_attn._videonsa_vision_mask = None

        return super().forward(*args, **kwargs)
