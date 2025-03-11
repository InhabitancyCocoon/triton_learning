import triton
import torch
import triton.language as tl
import tabulate

device = torch.device('cuda:0')
n_elements = 10
p = 0.3
BLOCK_SIZE = 64


@triton.jit
def naive_dropout_kernel(
    input_ptr,
    mask_ptr,
    output_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask)
    x_keep = tl.load(mask_ptr + offsets, mask)
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask)
    

def dropout_naive(input: torch.Tensor, mask: torch.Tensor, p: float) -> torch.Tensor:
    output = torch.empty_like(input)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    naive_dropout_kernel[grid](input, mask, output, n_elements, p, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def seeded_dropout_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask)


def dropout_seeded(input: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    output = torch.empty_like(input)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    seeded_dropout_kernel[grid](input, output, n_elements, p, seed, BLOCK_SIZE)
    return output


input = torch.rand(n_elements, device=device)
mask = (torch.rand(n_elements, device=device) > p).to(torch.int32)
output_triton_naive = dropout_naive(input, mask, p)
output_triton_seeded = dropout_seeded(input, p, 42)

print(
    tabulate.tabulate([
        ["input"] + input.tolist(),
        ["output_triton_naive"] + output_triton_naive.tolist(),
        ["output_triton_seeded"] + output_triton_seeded.tolist(),
    ]))
