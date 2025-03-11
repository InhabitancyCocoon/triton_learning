import triton
import torch
import triton.language as tl

device = torch.device('cuda:0')
M = 32
N = 512
p = 0.3
BLOCK_SIZE = 64


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
        output_row_ptr = output_ptr + row_idx * row_stride
        tl.store(output_row_ptr, row_output, mask)


def seeded_dropout(input: torch.Tensor, p: float, seed: torch.Tensor) -> torch.Tensor:
    assert input.is_contiguous()
    output = torch.empty_like(input)
    row_stride = input.stride(0)
    num_rows, num_cols = input.shape
    BLOCK_SIZE = triton.next_power_of_2(num_cols)

    



    return output



