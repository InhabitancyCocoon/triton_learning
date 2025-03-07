import torch
import triton
import triton.language as tl


device = torch.device("cuda:0")

@triton.jit
def swizzle_k(input_ptr, output_ptr, row_stride, group_size: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)

    pid_m_sw, pid_n_sw = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, group_size)

    offs_m = pid_m + tl.arange(0, 1)
    offs_n = pid_n + tl.arange(0, 1)
    offs = offs_m[:, None] * row_stride + offs_n[None, :]
    x = tl.load(input_ptr + offs)

    offs_m_sw = pid_m_sw + tl.arange(0, 1)
    offs_n_sw = pid_n_sw + tl.arange(0, 1)
    offs_sw = offs_m_sw[:, None] * row_stride + offs_n_sw[None, :]

    tl.store(output_ptr + offs_sw, x)


input = torch.arange(0, 5 * 7, device=device).view(5, 7)
print(input)
output = torch.empty_like(input)
print(output.shape, output.stride())
swizzle_k[(5, 7)](input, output, input.stride(0), 3)
print(output)