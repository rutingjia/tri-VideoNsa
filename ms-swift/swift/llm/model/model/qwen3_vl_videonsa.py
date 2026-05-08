# Copyright (c) Alibaba, Inc. and its affiliates.
# Qwen3-VL + VideoNSA decoder attention.
#
# This file does NOT modify the original Qwen3-VL implementation.
# It defines a new Qwen3-VL class and only replaces the language decoder
# self-attention modules.
#
# Intended location:
#   ms-swift/swift/llm/model/model/qwen3_vl_videonsa.py

import math
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


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def repeat_kv_for_videonsa(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads for GQA.

    Input:
        hidden_states: [batch, num_kv_heads, seq_len, head_dim]
    Output:
        hidden_states: [batch, num_attention_heads, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states

    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def _apply_attention_mask(
    scores: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if attention_mask is None:
        return scores

    if attention_mask.dtype == torch.bool:
        min_value = torch.finfo(scores.dtype).min
        return scores.masked_fill(~attention_mask, min_value)

    return scores + attention_mask


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

    If --attn_impl flash_attn is used by ms-swift, config._attn_implementation
    should be flash_attention_2, and this path will still dispatch to the
    HuggingFace flash attention interface.
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


def _build_videonsa_keep_mask(
    scores: torch.Tensor,
    video_token_mask: torch.Tensor,
    local_window: int,
    select_block_size: int,
    select_topk_blocks: int,
) -> torch.Tensor:
    """
    Build a VideoNSA-style sparse keep mask over raw K/V tokens.

    Pattern:
      - Text query tokens keep dense attention.
      - Video query tokens can attend to all text tokens.
      - Video query tokens attend to sparse video tokens:
          local window + selected top-k video blocks.

    Args:
        scores: [B, H, Q, K], already includes raw qk scores before softmax.
        video_token_mask: [B, L]
        local_window: local sequence window for video-video attention.
        select_block_size: block size used for top-k video block selection.
        select_topk_blocks: number of selected video blocks per query.

    Returns:
        keep_mask: [B, Q, K], bool.
    """
    batch, _, q_len, k_len = scores.shape
    device = scores.device

    video_token_mask = video_token_mask.to(device=device, dtype=torch.bool)

    q_is_video = video_token_mask[:, :, None]       # [B, Q, 1]
    k_is_video = video_token_mask[:, None, :]       # [B, 1, K]
    video_to_video = q_is_video & k_is_video        # [B, Q, K]

    # Text queries remain dense.
    # For video queries, all text keys remain available.
    keep_mask = (~q_is_video).expand(batch, q_len, k_len) | (~k_is_video).expand(batch, q_len, k_len)

    # Local video-video window.
    if local_window > 0:
        pos = torch.arange(q_len, device=device)
        local_keep = (pos[None, :, None] - pos[None, None, :]).abs() <= local_window
        local_keep = local_keep.expand(batch, q_len, k_len)
        keep_mask = keep_mask | (video_to_video & local_keep)

    # Selected global video blocks.
    if select_topk_blocks > 0 and select_block_size > 0:
        min_value = torch.finfo(scores.dtype).min

        # Average over heads for block routing score.
        routing_scores = scores.detach().mean(dim=1)  # [B, Q, K]
        routing_scores = routing_scores.masked_fill(~video_to_video, min_value)

        num_blocks = math.ceil(k_len / select_block_size)
        padded_len = num_blocks * select_block_size

        if padded_len > k_len:
            pad = padded_len - k_len
            routing_scores = torch.nn.functional.pad(routing_scores, (0, pad), value=min_value)

        block_scores = routing_scores.view(batch, q_len, num_blocks, select_block_size).amax(dim=-1)
        topk = min(select_topk_blocks, num_blocks)
        topk_block_idx = torch.topk(block_scores, k=topk, dim=-1).indices  # [B, Q, topk]

        block_keep = torch.zeros(
            batch,
            q_len,
            num_blocks,
            device=device,
            dtype=torch.bool,
        )
        block_keep.scatter_(dim=-1, index=topk_block_idx, value=True)

        key_block_ids = torch.arange(k_len, device=device) // select_block_size
        selected_keep = block_keep.gather(
            dim=-1,
            index=key_block_ids.view(1, 1, k_len).expand(batch, q_len, k_len),
        )

        keep_mask = keep_mask | (video_to_video & selected_keep)

    return keep_mask


def videonsa_decoder_attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    video_token_mask: Optional[torch.Tensor],
    dropout: float,
    scaling: float,
):
    """
    VideoNSA-style decoder attention.

    This is the Qwen3-VL decoder-side replacement. It keeps the same q/k/v/o
    projections and MRoPE path as Qwen3-VL, then sparsifies only video-to-video
    pairs in the language decoder sequence.

    Shapes:
        query_states: [B, H, Q, D]
        key_states:   [B, H_kv, K, D]
        value_states: [B, H_kv, K, D]

    Output:
        attn_output: [B, Q, H, D]
        attn_weights: None
    """
    batch, num_heads, q_len, _ = query_states.shape
    k_len = key_states.shape[2]

    # KV-cache generation path: keep original dense backend. This avoids
    # mismatched masks when Q=1 but K is long.
    if q_len != k_len:
        return None

    if video_token_mask is None or not torch.is_tensor(video_token_mask):
        return None

    if video_token_mask.ndim != 2:
        return None

    if video_token_mask.shape[0] != batch or video_token_mask.shape[1] != q_len:
        return None

    video_token_mask = video_token_mask.to(device=query_states.device, dtype=torch.bool)

    if not video_token_mask.any():
        return None

    num_key_value_groups = getattr(
        module,
        "num_key_value_groups",
        query_states.shape[1] // key_states.shape[1],
    )

    key_states = repeat_kv_for_videonsa(key_states, num_key_value_groups)
    value_states = repeat_kv_for_videonsa(value_states, num_key_value_groups)

    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    scores = _apply_attention_mask(scores, attention_mask)

    local_window = _env_int("VIDEONSA_LOCAL_WINDOW", 256)
    select_block_size = _env_int("VIDEONSA_SELECT_BLOCK_SIZE", 64)
    select_topk_blocks = _env_int("VIDEONSA_SELECT_TOPK_BLOCKS", 4)

    keep_mask = _build_videonsa_keep_mask(
        scores=scores,
        video_token_mask=video_token_mask,
        local_window=local_window,
        select_block_size=select_block_size,
        select_topk_blocks=select_topk_blocks,
    )

    min_value = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~keep_mask[:, None, :, :], min_value)

    attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = torch.dropout(attn_weights, p=dropout, train=module.training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


class Qwen3VLTextVideoNSAAttention(Qwen3VLTextAttention):
    """
    Drop-in replacement for Qwen3VLTextAttention.

    Parameter names stay identical to the original attention module:
      q_proj, k_proj, v_proj, o_proj, q_norm, k_norm

    Therefore pretrained Qwen3-VL weights can still load under strict=True.
    """

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
        video_token_mask = getattr(self, "_videonsa_video_token_mask", None)

        attn_output = None
        attn_weights = None

        if use_videonsa:
            result = videonsa_decoder_attention_forward(
                module=self,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                video_token_mask=video_token_mask,
                dropout=dropout,
                scaling=self.scaling,
            )
            if result is not None:
                attn_output, attn_weights = result

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
    Qwen3-VL with decoder-side VideoNSA attention.

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

    def _set_video_token_mask(self, video_token_mask: Optional[torch.Tensor]):
        for layer in self.model.language_model.layers:
            layer.self_attn._videonsa_video_token_mask = video_token_mask

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
                f"local_window={os.getenv('VIDEONSA_LOCAL_WINDOW', '256')}, "
                f"select_block_size={os.getenv('VIDEONSA_SELECT_BLOCK_SIZE', '64')}, "
                f"select_topk_blocks={os.getenv('VIDEONSA_SELECT_TOPK_BLOCKS', '4')}"
            )
            self._videonsa_printed = True

        self._set_video_token_mask(video_token_mask)

        try:
            return super().forward(*args, **kwargs)
        finally:
            self._set_video_token_mask(None)
