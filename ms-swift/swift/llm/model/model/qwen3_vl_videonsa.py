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
    Strict VideoNSA-style mixed decoder attention.

    hidden_states / q / k / v are mixed text+video tokens.

    Logic:
        1. Use vision_mask to split mixed query tokens into:
           - text query tokens
           - video query tokens

        2. Text branch:
           text queries + mixed keys/values -> original dense attention

        3. Video branch:
           video queries + mixed keys/values -> nsa_func

        4. Merge:
           output = text_output + video_output
    """

    batch_size, q_len, _ = hidden_states.shape
    k_len = key_states.shape[2]

    # Generation decode with KV cache: q_len is usually 1, k_len is long.
    # Keep original Qwen3-VL attention for now.
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

    # If no mask, fallback to original attention.
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

    vision_mask = vision_mask.to(device=hidden_states.device, dtype=torch.bool)

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

    # No video token: original dense attention.
    if not vision_mask.any():
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

    # ------------------------------------------------------------------
    # 1. Mixed token masks
    # ------------------------------------------------------------------
    text_mask = ~vision_mask

    # query_states shape: [B, H, L, D]
    # dense attention output shape: [B, L, H, D]
    text_query_mask = text_mask[:, None, :, None]      # [B, 1, L, 1]
    video_query_mask = vision_mask[:, None, :, None]   # [B, 1, L, 1]

    text_output_mask = text_mask[:, :, None, None]     # [B, L, 1, 1]
    video_output_mask = vision_mask[:, :, None, None]  # [B, L, 1, 1]

    # ------------------------------------------------------------------
    # 2. Text branch: text query tokens use original dense attention.
    #    K/V remain mixed, because text tokens still need to attend to
    #    previous video/text context.
    # ------------------------------------------------------------------
    text_query_states = query_states * text_query_mask

    text_output, dense_weights = _original_attention_forward(
        module=module,
        query_states=text_query_states,
        key_states=key_states,
        value_states=value_states,
        attention_mask=attention_mask,
        dropout=dropout,
        scaling=module.scaling,
        **kwargs,
    )
    # text_output: [B, L, H, D]
    text_output = text_output * text_output_mask

    # ------------------------------------------------------------------
    # 3. Video branch: video query tokens use NSA.
    #    Start from the same mixed q/k/v, but only video queries are active.
    # ------------------------------------------------------------------
    # Qwen3 states:
    #   query_states: [B, H, L, D]
    # NSA wants:
    #   q/k/v: [B, L, H, D]
    q = query_states.transpose(1, 2).contiguous()  # [B, L, H, D]
    k = key_states.transpose(1, 2).contiguous()    # [B, L, H_kv, D]
    v = value_states.transpose(1, 2).contiguous()  # [B, L, H_kv, D]

    q_video = q * video_output_mask

    # Gate projection for NSA branches.
    g = module.g_proj(hidden_states)
    g = g.view(batch_size, q_len, module.num_heads_for_nsa, 3)
    g_cmp, g_slc, g_swa = g.sigmoid().unbind(-1)
    # each: [B, L, H]

    # Only video query positions should contribute to NSA output.
    gate_video_mask = vision_mask[:, :, None]  # [B, L, 1]
    g_cmp = g_cmp * gate_video_mask
    g_slc = g_slc * gate_video_mask
    g_swa = g_swa * gate_video_mask

    block_size = _env_int("VIDEONSA_BLOCK_SIZE", 64)
    block_count = _env_int("VIDEONSA_BLOCK_COUNTS", 16)
    window_size = _env_int("VIDEONSA_WINDOW_SIZE", 512)

    cur_block_count = min(
        block_count,
        max(1, (q_len + block_size - 1) // block_size),
    )
    cur_window_size = min(window_size, q_len)

    video_output = nsa_func(
        q=q_video,
        k=k,
        v=v,
        g_cmp=g_cmp,
        g_slc=g_slc,
        g_swa=g_swa,
        block_count=cur_block_count,
        block_size=block_size,
        window_size=cur_window_size,
        scale=module.scaling,
        return_attn_weights=False,
        layer_idx=getattr(module, "layer_idx", None),
    )
    # video_output: [B, L, H, D]
    video_output = video_output * video_output_mask

    # ------------------------------------------------------------------
    # 4. Merge two branches back to mixed sequence.
    # ------------------------------------------------------------------
    mixed_output = text_output + video_output

    if _env_flag("VIDEONSA_DEBUG_SHAPES", "0") and not getattr(module, "_videonsa_debug_printed", False):
        print(
            "[Qwen3-VideoNSA][strict-mixed] "
            f"layer={getattr(module, 'layer_idx', None)}, "
            f"hidden={tuple(hidden_states.shape)}, "
            f"text_tokens={int(text_mask.sum().item())}, "
            f"video_tokens={int(vision_mask.sum().item())}, "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}, "
            f"text_output={tuple(text_output.shape)}, "
            f"video_output={tuple(video_output.shape)}, "
            f"mixed_output={tuple(mixed_output.shape)}"
        )
        module._videonsa_debug_printed = True

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
    # Prefer the mask created in the template before input_ids is replaced by inputs_embeds.
    # Shape should be [B, L], bool.
    video_token_mask = kwargs.pop("videonsa_vision_mask", None)

    # Fallback: if input_ids is still available, build the mask from video_token_id.
    input_ids = kwargs.get("input_ids", None)
    if video_token_mask is None:
        if input_ids is None and len(args) > 0 and torch.is_tensor(args[0]):
            input_ids = args[0]

        if input_ids is not None and torch.is_tensor(input_ids):
            video_token_mask = input_ids.eq(self._videonsa_video_token_id)

    # Make sure the mask is bool if it exists.
    if video_token_mask is not None:
        video_token_mask = video_token_mask.to(dtype=torch.bool)

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
