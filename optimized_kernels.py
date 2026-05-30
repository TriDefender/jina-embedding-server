"""
Optimized GPU kernels for jina-reranker-v3 (Qwen3-based).

Provides Triton flash attention kernel that replaces PyTorch's scaled_dot_product_attention
with a fused online-softmax implementation. Uses bf16 tensor cores for optimal throughput
on Ada Lovelace GPUs.

Usage:
    from optimized_kernels import patch_reranker_attention
    patch_reranker_attention(model)  # Call after model loading
"""

import math
import torch
import triton
import triton.language as tl


# =============================================================================
# Triton Flash Attention Kernel
# =============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    ],
    key=["M_size", "N_size", "D"],
)
@triton.jit
def _flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M_size, N_size,
    D: tl.constexpr,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused flash attention with online softmax. bf16 tensor cores, fp32 accumulation."""
    pid_z = tl.program_id(2)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(0)

    qkv_offset_z = pid_z * stride_qz
    qkv_offset_h = pid_h * stride_qh
    k_offset_z = pid_z * stride_kz
    k_offset_h = pid_h * stride_kh
    v_offset_z = pid_z * stride_vz
    v_offset_h = pid_h * stride_vh
    o_offset_z = pid_z * stride_oz
    o_offset_h = pid_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_ptrs = Q_ptr + qkv_offset_z + qkv_offset_h + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < M_size
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    if IS_CAUSAL:
        kv_end = tl.minimum(N_size, (pid_m + 1) * BLOCK_M)
    else:
        kv_end = N_size

    for start_n in range(0, kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + k_offset_z + k_offset_h + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = offs_n[:, None] < N_size
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # bf16 inputs → bf16 tensor cores (no TF32 truncation)
        qk = tl.dot(q, tl.trans(k))
        qk = qk.to(tl.float32) * sm_scale

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        kv_mask = offs_n[None, :] < N_size
        qk = tl.where(kv_mask, qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        exp_arg = qk - m_new[:, None]
        # NaN guard: replace NaN (from -inf - -inf) with -inf → exp → 0
        exp_arg = tl.where(exp_arg == exp_arg, exp_arg, float("-inf"))
        p = tl.exp(exp_arg)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V_ptr + v_offset_z + v_offset_h + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_mask = offs_n[:, None] < N_size
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # p in fp32, v promoted to fp32 → fp32 accumulation
        acc += tl.dot(p, v.to(tl.float32))
        m_i = m_new

    # Safe normalization (handles fully-masked rows)
    safe_l_i = tl.where(l_i[:, None] > 0, l_i[:, None], 1.0)
    acc = acc / safe_l_i
    acc = tl.where(l_i[:, None] > 0, acc, 0.0)

    o_ptrs = O_ptr + o_offset_z + o_offset_h + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o_mask = offs_m[:, None] < M_size
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=o_mask)


def triton_flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
    sm_scale: float = None,
) -> torch.Tensor:
    """Triton flash attention — drop-in replacement for scaled_dot_product_attention.

    Args:
        Q: [batch, heads, seq_len, head_dim] bf16/fp16
        K: [batch, kv_heads, seq_len, head_dim] bf16/fp16
        V: [batch, kv_heads, seq_len, head_dim] bf16/fp16
        causal: whether to apply causal masking
        sm_scale: softmax scale (default: 1/sqrt(head_dim))

    Returns:
        O: [batch, heads, seq_len, head_dim] same dtype as Q
    """
    Z, H, M_size, D = Q.shape
    _, _, N_size, _ = K.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    O = torch.empty_like(Q)

    def grid(META):
        return (triton.cdiv(M_size, META["BLOCK_M"]), H, Z)

    _flash_attention_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        Z, H, M_size, N_size,
        D=D,
        sm_scale=sm_scale,
        IS_CAUSAL=causal,
    )

    return O


# =============================================================================
# Model Patching
# =============================================================================


def _make_triton_attention_forward(attn_module):
    """Create a replacement forward for Qwen3Attention that uses Triton flash attention.

    Preserves: Q-norm, K-norm, RoPE, output projection.
    Replaces: scaled_dot_product_attention → triton_flash_attention.
    """
    q_proj = attn_module.q_proj
    k_proj = attn_module.k_proj
    v_proj = attn_module.v_proj
    o_proj = attn_module.o_proj
    q_norm = attn_module.q_norm
    k_norm = attn_module.k_norm
    head_dim = attn_module.head_dim
    num_heads = attn_module.config.num_attention_heads
    num_kv_heads = attn_module.config.num_key_value_heads

    # Import RoPE from transformers
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    def new_forward(hidden_states, position_embeddings, attention_mask=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)

        query_states = q_norm(q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = k_norm(k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # GQA: repeat KV heads to match query heads
        if num_heads != num_kv_heads:
            repeat = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(repeat, dim=1)
            value_states = value_states.repeat_interleave(repeat, dim=1)

        # Triton flash attention (causal=True for decoder)
        attn_output = triton_flash_attention(
            query_states, key_states, value_states,
            causal=True,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = o_proj(attn_output)
        return attn_output, None

    return new_forward


def patch_reranker_attention(model):
    """Patch all attention layers in a JinaForRanking model with Triton flash attention.

    Call this after loading the model and before inference.
    Compatible with CUDA Graph capture (the patched forward is a plain function).

    Args:
        model: JinaForRanking model (or any Qwen3-based model)
    """
    patched = 0
    for name, module in model.named_modules():
        if type(module).__name__ == "Qwen3Attention":
            module.forward = _make_triton_attention_forward(module)
            patched += 1

    if patched > 0:
        print(f"      [OK] Patched {patched} attention layers with Triton flash attention")
    else:
        print("      [WARN] No Qwen3Attention layers found to patch")

    return patched
