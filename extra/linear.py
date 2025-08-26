import torch
import triton
import triton.language as tl
from triton.runtime import driver
import pytest

torch.manual_seed(43)

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


@triton.autotune(
        configs=get_cuda_autotune_config(),
        key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    M, K, N,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    for step in range(tl.cdiv(K, BLOCK_SIZE_K)):
        
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - step * BLOCK_SIZE_K, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - step * BLOCK_SIZE_K, other=0.0).to(tl.float32)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.autotune(
        configs=get_cuda_autotune_config(),
        key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    M, K, N,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    for step in range(tl.cdiv(K, BLOCK_SIZE_K)):
        
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - step * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - step * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    accumulator += bias[None, :]

    accumulator = accumulator.to(tl.float16)

    # print(f"bias shape {bias.shape}, acc shape {accumulator.shape}")

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)



@triton.jit
def _bias_bwd(
    DY,
    DB,
    M,
    out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)

    db = tl.zeros((BLOCK_SIZE_OUT, ), dtype=tl.float32)

    for cur_row in range(0, M, BLOCK_SIZE_M):
        rows = cur_row + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < out_features)
        offs = rows[:, None] * out_features + cols[None, :]
        dy = tl.load(DY + offs, mask=mask, other=0.).to(tl.float32)
        db += tl.sum(dy, axis=0)

    tl.store(DB + cols, db, mask=cols < out_features)




class Linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias):
        M, in_features = x.shape
        assert in_features == weight.shape[1]
        out_features, _ = weight.shape
        y = torch.empty((M, out_features), device=x.device, dtype=x.dtype)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(out_features, META['BLOCK_SIZE_N']), )

        linear_kernel[grid](
            x,
            weight,
            y,
            bias,
            x.stride(0), x.stride(1),
            weight.stride(1), weight.stride(0),
            y.stride(0), y.stride(1),
            M, in_features, out_features,
        )

        ctx.save_for_backward(x, weight, bias)

        return y



    @staticmethod
    def backward(ctx, dy):
        x, w, b = ctx.saved_tensors
        M, in_features = x.shape
        out_features, _ = w.shape

        dx = torch.empty_like(x)
        dw = torch.empty_like(w)
        db = torch.empty_like(b)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(in_features, META['BLOCK_SIZE_N']), )
        # dx = dy @ w
        matmul_kernel[grid](
            dy,
            w,
            dx,
            dy.stride(0), dy.stride(1),
            w.stride(0), w.stride(1),
            dx.stride(0), dx.stride(1),
            M, out_features, in_features,
        )


        grid = lambda META: (triton.cdiv(out_features, META['BLOCK_SIZE_M']) * triton.cdiv(in_features, META['BLOCK_SIZE_N']), )
        # dw = dyT @ x
        matmul_kernel[grid](
            dy,
            x,
            dw,
            dy.stride(1), dy.stride(0),
            x.stride(0), x.stride(1),
            dw.stride(0), dw.stride(1),
            out_features, M, in_features,
        )


        # db = reduce(dy, dim=0)
        _bias_bwd[(triton.cdiv(out_features, 128), )](
            dy,
            db,
            M,
            out_features,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_OUT=128,
        )

        return dx, dw, db



linear = Linear.apply


@pytest.mark.parametrize(
    "M, in_features, out_features, dtype",
    [
        [652, 256, 512, torch.float16],
    ]
)
def test_linear(M, in_features, out_features, dtype):
    device = "cuda"
    
    x_shape = (M, in_features)
    w_shape = (out_features, in_features)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    weight = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(out_features, dtype=dtype, device=device, requires_grad=True)

    x.requires_grad_(True)

    y_torch = torch.nn.functional.linear(x, weight=weight, bias=bias)
    y_triton = linear(x, weight, bias)

    torch.testing.assert_close(y_triton, y_torch, atol=1e-4, rtol=1e-4)
    print("Congratulations, triton linear forward works!")




    dy = torch.randn(M, out_features, device=device, dtype=dtype)

    y_torch.backward(dy, retain_graph=True)
    dx_torch, dw_torch, db_torch = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None

    y_triton.backward(dy, retain_graph=True)
    dx_triton, dw_triton, db_triton = [_.grad.clone() for _ in [x, weight, bias]]

    torch.testing.assert_close(dx_triton, dx_torch, atol=1e-4, rtol=1e-4)
    print("Congratulations, triton linear x backward works!")


    torch.testing.assert_close(dw_triton, dw_torch, atol=1e-4, rtol=1e-4)
    print("Congratulations, triton linear weight backward works!")


    torch.testing.assert_close(db_triton, db_torch, atol=1e-4, rtol=1e-4)
    print("Congratulations, triton linear bias backward works!")




