# model/bi-lstm_model.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model

# --- Keras 3 compatible registration (with fallback) ---
try:
    # Keras 3
    from keras.saving import register_keras_serializable  # type: ignore
except Exception:  # pragma: no cover
    # tf.keras fallback
    from tensorflow.keras.utils import register_keras_serializable  # type: ignore


@register_keras_serializable(package="ra2")
class HybridPositionalEncoding(tf.keras.layers.Layer):
    """
    Hybrid token encoding for:
      - ERP tokens: (n_time,)
      - TFR tokens: (n_time * n_freq), ordered time-major then freq

    Adds:
      - time embedding for all tokens
      - type embedding (ERP vs TFR)
      - freq embedding for TFR tokens only
    """

    def __init__(self, n_time: int, n_freq: int, d_model: int, n_tokens_erp: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.n_time = int(n_time)
        self.n_freq = int(n_freq)
        self.d_model = int(d_model)
        self.n_tokens_erp = int(n_tokens_erp)

        # learnable embeddings
        self.time_emb = self.add_weight(
            shape=(self.n_time, self.d_model),
            initializer="random_normal",
            trainable=True,
            name="time_emb",
        )
        self.freq_emb = self.add_weight(
            shape=(self.n_freq, self.d_model),
            initializer="random_normal",
            trainable=True,
            name="freq_emb",
        )
        self.type_emb = self.add_weight(
            shape=(2, self.d_model),  # 0=ERP, 1=TFR
            initializer="random_normal",
            trainable=True,
            name="type_emb",
        )

        # Token index maps (static)
        n_tokens_tfr = self.n_time * self.n_freq
        n_tokens = self.n_tokens_erp + n_tokens_tfr

        time_idx = np.zeros((n_tokens,), dtype=np.int32)
        freq_idx = np.zeros((n_tokens,), dtype=np.int32)
        type_idx = np.zeros((n_tokens,), dtype=np.int32)

        # ERP tokens
        for k in range(self.n_tokens_erp):
            time_idx[k] = k
            freq_idx[k] = 0
            type_idx[k] = 0

        # TFR tokens
        for k in range(n_tokens_tfr):
            kk = self.n_tokens_erp + k
            t = k // self.n_freq
            f = k % self.n_freq
            time_idx[kk] = t
            freq_idx[kk] = f
            type_idx[kk] = 1

        self._time_idx = tf.constant(time_idx, dtype=tf.int32)
        self._freq_idx = tf.constant(freq_idx, dtype=tf.int32)
        self._type_idx = tf.constant(type_idx, dtype=tf.int32)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: (B, tokens, d_model)
        t = tf.gather(self.time_emb, self._time_idx)  # (tokens, d_model)
        typ = tf.gather(self.type_emb, self._type_idx)  # (tokens, d_model)
        f = tf.gather(self.freq_emb, self._freq_idx)  # (tokens, d_model)

        # Only apply freq embedding to TFR tokens
        is_tfr = tf.cast(tf.equal(self._type_idx, 1), tf.float32)[:, tf.newaxis]
        pe = t + typ + is_tfr * f
        return x + pe[tf.newaxis, :, :]

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update(
            {
                "n_time": self.n_time,
                "n_freq": self.n_freq,
                "d_model": self.d_model,
                "n_tokens_erp": self.n_tokens_erp,
            }
        )
        return cfg


@register_keras_serializable(package="ra2")
class ParameterTokenLayer(tf.keras.layers.Layer):
    """
    Creates one learned query token per parameter, broadcast to batch.

    Output: (B, n_params, d_model)
    """

    def __init__(self, n_params: int, d_model: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.n_params = int(n_params)
        self.d_model = int(d_model)

    def build(self, input_shape: tf.TensorShape) -> None:
        self.param_tokens = self.add_weight(
            shape=(self.n_params, self.d_model),
            initializer="random_normal",
            trainable=True,
            name="param_tokens",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        b = tf.shape(x)[0]
        return tf.tile(self.param_tokens[tf.newaxis, :, :], [b, 1, 1])

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"n_params": self.n_params, "d_model": self.d_model})
        return cfg


def transformer_encoder_block(
    x: tf.Tensor,
    d_model: int,
    num_heads: int,
    ff_dim: int,
    dropout_rate: float,
    block_id: int = 0,
) -> tf.Tensor:
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
        output_shape=d_model,
        name=f"self_attn_{block_id}",
    )(x, x)

    attn = Dropout(dropout_rate, name=f"self_attn_drop_{block_id}")(attn)
    x = Add(name=f"self_attn_add_{block_id}")([x, attn])
    x = LayerNormalization(epsilon=1e-6, name=f"self_attn_norm_{block_id}")(x)

    ff = Dense(ff_dim, activation="gelu", name=f"ff1_{block_id}")(x)
    ff = Dense(d_model, name=f"ff2_{block_id}")(ff)
    ff = Dropout(dropout_rate, name=f"ff_drop_{block_id}")(ff)

    x = Add(name=f"ff_add_{block_id}")([x, ff])
    x = LayerNormalization(epsilon=1e-6, name=f"ff_norm_{block_id}")(x)
    return x


def build_bi_lstm_model(
    n_tokens: int,
    feature_dim: int,
    n_params: int,
    n_time_patches: int,
    n_freq_patches: int,
    n_tokens_erp: int,
    d_model: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    ff_dim: int = 256,
    dropout_rate: float = 0.1,
    return_attention: bool = False,
) -> Model:
    """
    Transformer encoder + per-parameter query tokens.

    Outputs
    -------
    If return_attention=False:
        Tensor (B, 2P): concatenated [mu_z, logvar_z]
    If return_attention=True:
        [pred, scores] where scores is attention weights
    """
    inp = Input(shape=(n_tokens, feature_dim), name="tokens_input")

    x = Dense(d_model, name="token_proj")(inp)
    x = HybridPositionalEncoding(
        n_time=n_time_patches,
        n_freq=n_freq_patches,
        d_model=d_model,
        n_tokens_erp=n_tokens_erp,
        name="hybrid_posenc",
    )(x)

    for li in range(num_layers):
        x = transformer_encoder_block(
            x,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            block_id=li,
        )

    q = ParameterTokenLayer(n_params, d_model, name="param_tokens")(x)

    mha = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
        output_shape=d_model,
        name="param_cross_attn",
    )

    if return_attention:
        ctx, scores = mha(q, x, return_attention_scores=True)
    else:
        ctx = mha(q, x)
        scores = None

    ctx = Dropout(dropout_rate, name="param_ctx_drop")(ctx)
    ctx = LayerNormalization(epsilon=1e-6, name="param_ctx_norm")(ctx)

    h = Dense(ff_dim, activation="gelu", name="param_head_fc")(ctx)
    h = Dropout(dropout_rate, name="param_head_drop")(h)
    out2 = Dense(2, activation="linear", name="param_mu_logvar")(h)  # (B, P, 2)

    mu_z = out2[:, :, 0]
    logvar_z = out2[:, :, 1]

    pred = Concatenate(axis=1, name="pred_mu_logvar_z")([mu_z, logvar_z])

    if return_attention:
        return Model(inp, [pred, scores], name="JR_Transformer_ParamToken_HybridAttn")
    return Model(inp, pred, name="JR_Transformer_ParamToken_Hybrid")


def get_custom_objects() -> Dict[str, Any]:
    """
    Returns custom object mapping for robust Keras model loading.

    We include both plain class names and the registered names that appear in
    saved configs (e.g., 'ra2>HybridPositionalEncoding').
    """
    return {
        "HybridPositionalEncoding": HybridPositionalEncoding,
        "ParameterTokenLayer": ParameterTokenLayer,
        "ra2>HybridPositionalEncoding": HybridPositionalEncoding,
        "ra2>ParameterTokenLayer": ParameterTokenLayer,
    }
