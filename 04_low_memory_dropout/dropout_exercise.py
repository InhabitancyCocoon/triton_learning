import triton
import torch
import triton.language as tl
from triton.runtime import driver

torch.manual_seed(42)
device = torch.cuda.current_device()
M = 32
N = 512
p = 0.3
BLOCK_SIZE = 64

properties = driver.active.utils.get_device_properties(device)
print(properties)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]


# assume the row can fit into GPU SRAM, each program is reponsible for multiple rows.
@triton.jit
def seeded_dropout_kernel(
    input_ptr,
    output_ptr,
    p,
    seed_ptr,
    row_stride,
    num_rows,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, num_rows, row_step, num_stages):
        input_row_ptr = input_ptr + row_idx * row_stride
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_cols
        row = tl.load(input_row_ptr + offsets, mask)
        seed = tl.load(seed_ptr + row_idx)
        keep_mask = tl.rand(offsets, seed) > p
        row_output = tl.where(keep_mask, row / (1 - p), 0.0)
        output_row_start_ptr = output_ptr + row_idx * row_stride
        tl.store(output_row_start_ptr + offsets, row_output, mask)


def seeded_dropout(input: torch.Tensor, p: float, seed: torch.Tensor) -> torch.Tensor:
    assert input.is_contiguous()
    output = torch.empty_like(input)
    row_stride = input.stride(0)
    num_rows, num_cols = input.shape
    BLOCK_SIZE = triton.next_power_of_2(num_cols)

    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200_000 else 2

    kernel = seeded_dropout_kernel.warmup(
        input, output, p, seed, row_stride, num_rows, num_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,)
    )
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)

    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, num_rows)

    kernel[(num_programs, 1, 1)](input, output, p, seed, row_stride, num_rows, num_cols, BLOCK_SIZE, num_stages)

    return output



input = torch.rand(M, N, device=device)
seed = torch.empty_like(input)
torch.fill(seed, 42)
p = 0.3
output = seeded_dropout(input, p, seed)

input_elements = input.numel()
expected_elements = int(input_elements * (1 - p))
output_elements = torch.sum(output != 0)
print(input_elements, expected_elements, output_elements)
