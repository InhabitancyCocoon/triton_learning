import triton
import triton.language as tl
import torch
from typing import Optional



@triton.jit
def _tma_fill_kernel(a_ptr):
    desc = tl.make_tensor_descriptor(
        a_ptr,
        [256, 256],
        [256, 1],
        block_shape=[16, 16]
    )

    tid_m = tl.program_id(0)
    tid_n = tl.program_id(1)

    m_offset = tid_m * 16
    n_offset = tid_n * 16

    values = desc.load([m_offset, n_offset])
    desc.store([m_offset, n_offset], values * 2)


def tma_fill():
    a = torch.randn(256, 256, device="cuda")

    def alloc_func(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.uint8)

    print(f"before tma fill: sum(a) = {a.sum()}")

    triton.set_allocator(alloc_func)

    block_size_m = block_size_n = 16
    _tma_fill_kernel[(16, 16)](a)

    print(f"after tma fill: sum(a) = {a.sum()}")


if __name__ == "__main__":
    tma_fill()