"""Regla delta con compuerta (GatedDeltaNet de Qwen3.5) por BLOQUES, diferenciable, para entrenar con mlx-lm.

En entrenamiento mlx-lm usa la recurrencia paso a paso (gated_delta_ops), que para el backward guarda un estado
[H, Dv, Dk] por TOKEN: con prompts de ~3000 tokens son varios GB por capa. Aquí se procesa por bloques de C tokens
(algoritmo "chunk" de flash-linear-attention, representación WY): sólo se guarda un estado por bloque.

    S_t = a_t S_{t-1} + b_t (v_t - a_t S_{t-1} k_t) k_t^T ,   y_t = S_t q_t

`install()` sustituye gated_delta_update en qwen3_5 sólo cuando se llama con use_kernel=False (entrenamiento).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

CHUNK = 64
INV_BLOCK = 8


def _inv_doubling(L: mx.array, n: int) -> mx.array:
    """(I + L)^-1 = prod_j (I + (-L)^(2^j)) para L estrictamente inferior; estable sólo en bloques pequeños."""
    eye = mx.eye(n, dtype=L.dtype)
    N = -L
    out = eye + N
    P = N
    k = 2
    while k < n:
        P = P @ P
        out = out @ (eye + P)
        k *= 2
    return out


def _inv_unit_lower(L: mx.array, n: int, block: int = INV_BLOCK) -> mx.array:
    """(I + L)^-1 con L estrictamente triangular inferior: bloques diagonales de `block` por duplicación (pocas
    potencias, sin desbordar) y sustitución hacia delante por bloques entre ellos."""
    if n <= block:
        return _inv_doubling(L, n)
    nb = n // block
    sl = lambda i: slice(i * block, (i + 1) * block)  # noqa: E731
    diag = [_inv_doubling(L[..., sl(i), sl(i)], block) for i in range(nb)]
    X = [[None] * nb for _ in range(nb)]
    for i in range(nb):
        X[i][i] = diag[i]
        for j in range(i - 1, -1, -1):
            acc = sum(L[..., sl(i), sl(k)] @ X[k][j] for k in range(j, i))
            X[i][j] = -(diag[i] @ acc)
    zero = mx.zeros(L.shape[:-2] + (block, block), dtype=L.dtype)
    return mx.concatenate([mx.concatenate([X[i][j] if j <= i else zero for j in range(nb)], axis=-1)
                           for i in range(nb)], axis=-2)


@mx.custom_function
def inv_unit_lower(L: mx.array) -> mx.array:
    return _inv_unit_lower(L, L.shape[-1])


@inv_unit_lower.vjp
def _inv_vjp(primals, cotangent, output):
    """X = (I + L)^-1  ⇒  dX = -X dL X  ⇒  dL = -(X^T G X^T), sólo la parte estrictamente inferior."""
    X = output
    Xt = X.swapaxes(-1, -2)
    n = X.shape[-1]
    strict = mx.tril(mx.ones((n, n), dtype=mx.bool_), k=-1)
    return (mx.where(strict, -(Xt @ cotangent @ Xt), 0.0),)


def chunk_gated_delta(q, k, v, g_log, beta, state, chunk: int = CHUNK):
    """q, k: [B, T, H, Dk]; v: [B, T, H, Dv]; g_log, beta: [B, T, H]; state: [B, H, Dv, Dk] (o None).
    Devuelve y [B, T, H, Dv] y el estado final [B, H, Dv, Dk] (mismas convenciones que gated_delta_ops)."""
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    dtype = q.dtype
    pad = (-T) % chunk
    if pad:
        z = lambda x: mx.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))  # noqa: E731
        q, k, v, g_log, beta = z(q), z(k), z(v), z(g_log), z(beta)
    n = (T + pad) // chunk
    f32 = mx.float32
    # [B, H, n, C, *]
    tr = lambda x: x.astype(f32).reshape(B, n, chunk, H, -1).transpose(0, 3, 1, 2, 4)  # noqa: E731
    q, k, v = tr(q), tr(k), tr(v)
    g = g_log.astype(f32).reshape(B, n, chunk, H).transpose(0, 3, 1, 2)  # [B,H,n,C]
    b = beta.astype(f32).reshape(B, n, chunk, H).transpose(0, 3, 1, 2)
    gc = mx.cumsum(g, axis=-1)  # decaimiento acumulado dentro del bloque
    diff = gc[..., :, None] - gc[..., None, :]
    lower = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_))
    strict = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_), k=-1)
    decay_mat = mx.where(lower, mx.exp(mx.where(lower, diff, 0.0)), 0.0)  # Γ_ij = exp(g_i - g_j), i >= j
    kb = k * b[..., None]
    Lm = mx.where(strict, (kb @ k.swapaxes(-1, -2)) * decay_mat, 0.0)
    Tm = inv_unit_lower(Lm)  # (I + L)^-1
    U = Tm @ (v * b[..., None])  # [B,H,n,C,Dv]
    W = Tm @ (kb * mx.exp(gc)[..., None])  # [B,H,n,C,Dk]
    qk = mx.where(lower, (q @ k.swapaxes(-1, -2)) * decay_mat, 0.0)  # incluye la diagonal
    S = (mx.zeros((B, H, Dv, Dk), dtype=f32) if state is None else state.astype(f32))
    ys = []
    for i in range(n):
        St = S.swapaxes(-1, -2)  # [B,H,Dk,Dv]
        v_new = U[:, :, i] - W[:, :, i] @ St
        o = (q[:, :, i] * mx.exp(gc[:, :, i])[..., None]) @ St + qk[:, :, i] @ v_new
        ys.append(o)
        last = gc[:, :, i, -1]
        kd = k[:, :, i] * mx.exp(last[..., None] - gc[:, :, i])[..., None]  # [B,H,C,Dk]
        S = S * mx.exp(last)[..., None, None] + v_new.swapaxes(-1, -2) @ kd  # [B,H,Dv,Dk]
    y = mx.stack(ys, axis=2)  # [B,H,n,C,Dv]
    y = y.transpose(0, 2, 3, 1, 4).reshape(B, n * chunk, H, Dv)[:, :T]
    return y.astype(dtype), S


def install() -> None:
    from mlx_lm.models import gated_delta as gd
    from mlx_lm.models import qwen3_5

    original = gd.gated_delta_update

    def update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, *, use_kernel=True, lower_bound=None,
               allow_neg_eigval=False):
        if use_kernel or lower_bound is not None or mask is not None:
            return original(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel,
                            lower_bound=lower_bound, allow_neg_eigval=allow_neg_eigval)
        beta = mx.sigmoid(b)
        if allow_neg_eigval:
            beta = beta * 2.0
        g_log = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
        Hk, Hv = q.shape[-2], v.shape[-2]
        if Hv // Hk > 1:
            q = mx.repeat(q, Hv // Hk, -2)
            k = mx.repeat(k, Hv // Hk, -2)
        return chunk_gated_delta(q, k, v, g_log, beta, state)

    qwen3_5.gated_delta_update = update


if __name__ == "__main__":  # comprobación numérica contra la recurrencia de referencia
    from mlx_lm.models.gated_delta import gated_delta_ops, compute_g

    mx.random.seed(0)
    B, T, Hk, Hv, Dk, Dv = 1, 150, 2, 4, 32, 32
    q = mx.random.normal((B, T, Hk, Dk)) * 0.2
    k = mx.random.normal((B, T, Hk, Dk))
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    v = mx.random.normal((B, T, Hv, Dv))
    a = mx.random.normal((B, T, Hv))
    bb = mx.random.normal((B, T, Hv))
    A_log = mx.random.normal((Hv,)) * 0.5
    dt = mx.random.normal((Hv,)) * 0.5
    s0 = mx.random.normal((B, Hv, Dv, Dk)) * 0.1
    g = compute_g(A_log, a, dt)
    y_ref, s_ref = gated_delta_ops(q, k, v, g, mx.sigmoid(bb), s0)
    install()
    from mlx_lm.models import qwen3_5
    y, s = qwen3_5.gated_delta_update(q, k, v, a, bb, A_log, dt, s0, use_kernel=False)
    print("max |dy|", mx.abs(y - y_ref).max().item(), "max |ds|", mx.abs(s - s_ref).max().item(),
          "scale", mx.abs(y_ref).max().item())

    def f(q, k, v):
        return qwen3_5.gated_delta_update(q, k, v, a, bb, A_log, dt, s0, use_kernel=False)[0].sum()

    def f_ref(q, k, v):
        return gated_delta_ops(q, k, v, g, mx.sigmoid(bb), s0)[0].sum()

    g1 = mx.grad(f, argnums=(0, 1, 2))(q, k, v)
    g2 = mx.grad(f_ref, argnums=(0, 1, 2))(q, k, v)
    print("max |dgrad|", [round(mx.abs(x - y).max().item(), 6) for x, y in zip(g1, g2)])
