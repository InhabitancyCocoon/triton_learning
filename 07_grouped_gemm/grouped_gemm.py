import torch
import triton
import triton.language as tl


group_m = [1024, 512, 256, 128]
group_n = [1024, 512, 256, 128]
group_k = [1024, 512, 256, 128]

assert len(group_m) == len(group_n)
assert len(group_k) == len(group_n)

group_size = len(group_m)

group_A, group_B = [], []

for i in range(group_size):
    M, N, K = group_m[i], group_n[i], group_k[i]
    A = torch.rand((M, K), device='cuda', dtype=torch.bfloat16)
    B = torch.rand((N, K), device='cuda', dtype=torch.bfloat16)
    group_A.append(A)
    group_B.append(B)


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_gemm_kernel(
    A_ptrs,
    B_ptrs,
    C_ptrs,
    g_sizes,
    g_lds,
    group_size,
    NUM_SUM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(g_sizes + g * 3)
        gn = tl.load(g_sizes + g * 3 + 1)
        gk = tl.load(g_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = ?
            b_ptr = ?
            c_ptr = ?

            tile_idx_within = tile_idx - last_problem_end
            tile_m_idx = tile_idx_within // num_n_tiles
            tile_n_idx = tile_idx_within % num_n_tiles

            # do regular gemm here
            





def grouped_gemm_fn(group_A, group_B):
    device = torch.device('cuda')
    assert len(group_A) == len(group_B)

    A_ptrs = []
    B_ptrs = []
    C_ptrs = []
    g_sizes = []
    g_lds = []
    group_C = []

    for A, B in zip(group_A, group_B):
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=device, dtype=A.dtype)
        group_C.append(C)

        A_ptrs.append(A.data_ptr())
        B_ptrs.append(B.data_ptr())
        C_ptrs.append(C.data_ptr())

        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    
    A_ptrs = torch.tensor(A_ptrs, device=device)
    B_ptrs = torch.tensor(B_ptrs, device=device)
    C_ptrs = torch.tensor(C_ptrs, device=device)
    g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)

    grid = lambda META: (META['NUM_SUM'], )

    grouped_gemm_kernel[grid](
        A_ptrs,
        B_ptrs,
        C_ptrs,
        g_sizes,
        g_lds,
        group_size
    )

    return group_C



ref_out = [torch.matmul(A, B) for A, B in zip(group_A, group_B)]
tri_out = None


for i in range(group_size):
    torch.testing.assert_close(ref_out[i], tri_out[i], atol=1e-2, rtol=0)