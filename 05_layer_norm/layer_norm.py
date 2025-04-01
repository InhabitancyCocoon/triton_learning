import torch
import triton
import triton.language as tl
from triton.runtime import driver

torch.manual_seed(42)


@triton.jit
def _layer_norm_fwd_fused(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        _mean += tl.load(X + cols, cols < N, other=0.0).to(tl.float32)
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        a = tl.where(cols < N, a - mean, 0.0)
        _var += a * a
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    tl.store(Mean + row, mean)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = (x - mean) * rstd
        w = tl.load(W + cols, mask=cols < N)
        b = tl.load(B + cols, mask=cols < N)
        y = x * w + b
        tl.store(Y + cols, y, mask=cols < N)


@triton.jit
def _layer_norm_bwd_dx_fused(
    DX,  # pointer to the input gradient to be filled
    DY,  # pointer to the output gradient
    DW,  # pointer to the partial sum of the weight gradients to be filled
    DB,  # pointer to the partial sum of the bias gradients to be filled
    X,
    W,
    Mean,
    Rstd,
    Lock,  # pointer to the lock
    stride,
    N,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride

    lock_id = row % GROUP_SIZE_M
    Lock += lock_id

    Count = Lock + GROUP_SIZE_M



    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols

    # load data to the SRAM
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    # compute dx, a little bit complex
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(wdy, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd

    # store dx
    tl.store(DX + cols, dx, mask=mask)

    # accumulate to partial_dw and partial_db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)

    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass

    count = tl.load(Count)

    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(
    DW,
    DB,
    FINAL_DW,
    FINAL_DB,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask, other=0.)
        db += tl.load(DB + offs, mask, other=0.)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)



class LayerNorm(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        y = torch.empty_like(x)

        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device='cuda')
        rstd = torch.empty((M, ), dtype=torch.float32, device='cuda')

        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layernorm doesn't support feature dim >= 64KB")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        _layer_norm_fwd_fused[(M, )](
            x_arg, y, weight, bias, mean, rstd,
            x_arg.stride(0), N, eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y
        

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, r = ctx.saved_tensors
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192:
            GROUP_SIZE_M = 96
        if N <= 4096:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device='cuda')
        _dw = torch.empty((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        _db = torch.empty((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)

        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_dx_fused[(M, )](
            dx, dy, _dw, _db, x, w, m, r, locks,
            x_arg.stride(0), N,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=ctx.num_warps
        )
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        _layer_norm_bwd_dwdb[grid](
            _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128,
            num_ctas=1,
        )
        return dx, None, dw, db, None
    

layer_norm = LayerNorm.apply


def test_layer_norm(M, N, dtype, eps=1e-5, device='cuda'):
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)

    # the constant here leads to right result
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)

    y_triton = layer_norm(x, w_shape, weight, bias, eps)
    y_torch = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype=dtype)

    y_triton.backward(dy, retain_graph=True)
    dx_triton, dw_triton, db_triton = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None

    y_torch.backward(dy, retain_graph=True)
    dx_torch, dw_torch, db_torch = [_.grad.clone() for _ in [x, weight, bias]]

    torch.testing.assert_close(y_triton, y_torch, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(dx_triton, dx_torch, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(dw_triton, dw_torch, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(db_triton, db_torch, atol=1e-2, rtol=1e-2)



test_layer_norm(1151, 8192, torch.float16)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name="layernorm backward",
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'}
    )
)
def bench_layer_norm(M, N, dtype, provider, mode='backward', eps=1e-5, device='cuda'):
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)

    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == 'triton':
            return layer_norm(x, w_shape, weight, bias, eps)

        if provider == 'torch':
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)
    


    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=200)

    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles, grad_to_none=[x],  rep=200)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


bench_layer_norm.run(save_path='./result', print_data=True)
