import triton
import triton.language as tl
import torch

torch.manual_seed(42)

N = 9876

device = torch.device('cuda')

x = torch.rand(N).to(device)
y = torch.rand(N).to(device)



@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements)
    y = tl.load(y_ptr + offsets, mask=offsets < n_elements)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=offsets < n_elements)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output



output_triton = add(x, y)
output_torch = x + y


torch.testing.assert_close(output_triton, output_torch)

