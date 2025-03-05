import torch
import triton
import triton.language as tl
from triton.runtime import driver

# The result is obtained on L20.
# Be careful of the version mismatch between the tutorial and your software.


torch.manual_seed(42)

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    row_max = torch.max(x, dim=1, keepdim=True).values
    z = x - row_max
    numerator = torch.exp(z)
    denominator = torch.sum(numerator, dim=1, keepdim=True)
    return numerator / denominator

M = 1823
N = 781

device = torch.cuda.current_device()

input = torch.rand(M, N, device=device)


# assume the row can fit into GPU SRAM, each program is reponsible for multiple rows.
@triton.jit
def softmax_kernel(x_ptr,
                   output_ptr,
                   row_stride,
                   num_rows,
                   num_cols,
                   BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr,
                   ):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, num_rows, row_step, num_stages=num_stages):
        row_start_ptr = x_ptr + row_idx * row_stride
        offsets = tl.arange(0, BLOCK_SIZE)
        row = tl.load(row_start_ptr + offsets, mask=offsets < num_cols, other=-float('inf'))
        row_stable = row - tl.max(row, axis=0)
        numerator = tl.exp(row_stable)
        denominator = tl.sum(numerator, axis=0)
        output_softmax = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * row_stride
        tl.store(output_row_start_ptr + offsets, value=output_softmax, mask=offsets < num_cols)



properties = driver.active.utils.get_device_properties(device)
print(properties)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]



def softmax(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    num_rows, num_cols = x.shape
    row_stride = x.stride(0)
    n_elements = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(num_cols)


    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200_000 else 2

    kernel = softmax_kernel.warmup(x, output, row_stride, num_rows, num_cols, 
                                   BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, 
                                   num_warps=num_warps, grid=(1,))
    kernel._init_handles()

    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)

    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, num_rows)


    kernel[(num_programs, 1, 1)](x, output, row_stride, num_rows, num_cols)

    return output



output_naive = naive_softmax(input)
output_torch = torch.softmax(input, dim=1)
output_triton = softmax(input)

torch.testing.assert_close(output_torch, output_naive)
torch.testing.assert_close(output_torch, output_triton)

print("The result is correct")
