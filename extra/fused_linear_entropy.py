"""

TODO:

cutlass, cute
fp8, wait, it doesn't seem rational to compute cross entropy with fp8 precision.

(input, accumulate, forward loss)
(fp32, fp32, fp32)
(bf16, fp32, fp32)

ref link:

https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py

https://github.com/mgmalek/efficient_cross_entropy/blob/main/test.py

https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

https://discuss.pytorch.org/t/the-difference-between-torch-tensor-data-and-torch-tensor/25995

https://docs.pytorch.org/docs/stable/generated/torch.gather.html

https://www.cnblogs.com/zzk0/p/15173022.html

https://docs.pytorch.org/docs/stable/amp.html

https://github1s.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py

"""

import torch
import pytest
import itertools
import triton
import triton.language as tl
from torch.nn import functional as F


# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2


@triton.jit
def _entropy_fwd_bwd_kernel(
    logit_ptr,
    label_ptr,
    loss_ptr,
    scale,
    logit_row_stride,
    logit_col_stride,
    label_stride,
    loss_stride,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    logit_ptr += pid * logit_row_stride
    label_ptr += pid * label_stride
    loss_ptr += pid * loss_stride

    label = tl.load(label_ptr)
    orig_label_logit = tl.load(logit_ptr + label * logit_col_stride).to(tl.float32)

    max_logit = float("-inf")
    deno = 0.0

    # 1. online chunkwise softmax
    for i in range(0, VOCAB_SIZE, BLOCK_SIZE_N):
        chunk_logit_col_offset = tl.arange(0, BLOCK_SIZE_N) + i
        chunk_logit_col_mask = chunk_logit_col_offset < VOCAB_SIZE
        chunk_logit = tl.load(logit_ptr + chunk_logit_col_offset * logit_col_stride,
                        mask=chunk_logit_col_mask,
                        other=float("-inf"),
                    ).to(tl.float32)  # cast, to ?
        
        new_max_logit = tl.maximum(max_logit, tl.max(chunk_logit))
        cur_deno = tl.sum(tl.exp(chunk_logit - new_max_logit))
        deno = deno * tl.exp(max_logit - new_max_logit) + cur_deno
        max_logit = new_max_logit

    lse = max_logit + tl.log(deno)
    # 2. compute the cross entropy loss for each token
    loss = lse - orig_label_logit  # some smart math...
    tl.store(loss_ptr, loss)

    for i in range(0, VOCAB_SIZE, BLOCK_SIZE_N):
        chunk_logit_col_offset = tl.arange(0, BLOCK_SIZE_N) + i
        chunk_logit_col_mask = chunk_logit_col_offset < VOCAB_SIZE
        chunk_logit = tl.load(logit_ptr + chunk_logit_col_offset * logit_col_stride,
                        mask=chunk_logit_col_mask,
                        other=float("-inf"),
                    ).to(tl.float32)  # cast, to ?
        
        chunk_softmax = tl.exp(chunk_logit - max_logit) / deno

        # 3. update the logit in-place (scatter_add_ and mul)
        # I wonder why liger kernel choose to launch another element mul kernel for scale during the backward pass.
        chunk_grad_softmax = tl.where(chunk_logit_col_offset != label, chunk_softmax, chunk_softmax - 1) * scale
        tl.store(logit_ptr + chunk_logit_col_offset * logit_col_stride, chunk_grad_softmax, mask=chunk_logit_col_mask)

    
    

class FusedLinearEntropy(torch.autograd.Function):
    """
    Fuse linear and cross entropy together.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        label: torch.Tensor,
        reduction: str,
    ):
        assert reduction in ["mean", "sum"], "Supported reduction must be: " \
                                                     "mean, sum"

        # Optimization 2: as we always return a scalar, we can compute the final result step by step.
        # Here I just copy the chunk logic from liger kernel.

        num_tokens, H = input.shape
        vocab_size = weight.shape[0]

        if reduction == "mean":
            scale = 1.0 / num_tokens
        elif reduction == "sum":
            scale = 1.0

        inc_factor = triton.cdiv(vocab_size, H)  # (V + H - 1) // H
        chunk_size = triton.next_power_of_2(triton.cdiv(num_tokens, inc_factor))  # (BT + inc_factor - 1) // inc_factor
        num_chunks = triton.cdiv(num_tokens, chunk_size)  # (BT + chunk_size - 1) // chunk_size

        loss_per_token = torch.empty(num_tokens, dtype=torch.float32, device=input.device)
        grad_input = torch.empty_like(input)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        grad_bias = torch.zeros_like(bias)

        for chunk_id in range(num_chunks):
            start_token_idx = chunk_id * chunk_size
            end_token_idx = min((chunk_id + 1) * chunk_size, num_tokens)
            chunk_num_tokens = end_token_idx - start_token_idx

            chunk_input = input[start_token_idx : end_token_idx]
            chunk_label = label[start_token_idx : end_token_idx]
            chunk_loss = loss_per_token[start_token_idx : end_token_idx]
            chunk_logit = chunk_input @ weight.T + bias[None, :]

            

            # the below operations(chunk cross entropy fwd, bwd) can be fused into a triton kernel
            # 0. one triton program handles one token
            # 1. online softmax (fp32 precision), vocab_size can be large
            # 2. compute the cross entropy for each token
            # 3. update the logit in-place (scatter_add_ and mul)

            chunk_logit = chunk_logit.contiguous()
            chunk_label = chunk_label.contiguous()

            BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(vocab_size))

            # all chunk here...
            _entropy_fwd_bwd_kernel[(chunk_num_tokens,)](
                chunk_logit,
                chunk_label,
                chunk_loss,  # a stupid mistake
                scale,
                chunk_logit.stride(0),
                chunk_logit.stride(1),
                chunk_label.stride(0),
                chunk_loss.stride(0),
                vocab_size,
                BLOCK_SIZE_N,
            )

            chunk_grad_softmax = chunk_logit

            # chunk linear backward
            chunk_grad_softmax = chunk_grad_softmax.to(weight.dtype)
            chunk_grad_input = chunk_grad_softmax @ weight
            grad_input[start_token_idx : end_token_idx] = chunk_grad_input
            grad_weight += chunk_grad_softmax.T @ chunk_input
            grad_bias += torch.sum(chunk_grad_softmax, dim=0)

        ctx.num_tokens = num_tokens
        ctx.H = H
        ctx.vocab_size = vocab_size
        ctx.reduction = reduction
        
        ctx.save_for_backward(
            grad_input,
            grad_weight,
            grad_bias
        )  # should be called only once

        if reduction == "mean":
            return loss_per_token.mean()
        else:
            return loss_per_token.sum()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        return grad_input, grad_weight, grad_bias, None, None


# unhappy with keyword arguments
fused_torch_linear_entropy = FusedLinearEntropy.apply


def ref_torch_linear_entropy(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    label: torch.Tensor,
    reduction: str,
):
    logit = F.linear(input, weight, bias).float()
    entropy = F.cross_entropy(logit, label, reduction=reduction)
    return entropy


# pytest.mark.parametrize is unhappy with default param.
# the atol and rtol follows the liger kernel.
# https://github1s.com/linkedin/Liger-Kernel/blob/main/test/transformers/test_fused_linear_cross_entropy.py
@pytest.mark.parametrize(
    "B, SEQ, H, num_classes",
    itertools.product(
        [1, 8, 17, 33],
        [1, 4, 7, 11, 256, 257],
        [1, 3, 7, 16],
        [1, 255, 512, 1023],
    )
)
@pytest.mark.parametrize(
    "dtype, reduction, atol, rtol",
    [
        (torch.bfloat16, "mean", 5e-3, 5e-2),
        (torch.float32, "mean", 1e-5, 5e-4),
        (torch.bfloat16, "sum", 5e0, 5e0),
        (torch.float32, "sum", 1e-3, 5e-2),
    ],
)
def test_linear_entropy(B, SEQ, H, num_classes, dtype, reduction, atol, rtol):
    torch.manual_seed(43)

    ref_input = torch.rand(B * SEQ, H, requires_grad=True, device="cuda", dtype=dtype)
    ref_weight = torch.empty(num_classes, H, requires_grad=True, device="cuda", dtype=dtype)
    # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
    with torch.no_grad():
        ref_weight.uniform_()
    ref_bias = torch.rand(num_classes, requires_grad=True, device="cuda", dtype=dtype)

    input = ref_input.detach().clone().requires_grad_(True)
    weight = ref_weight.detach().clone().requires_grad_(True)
    bias = ref_bias.detach().clone().requires_grad_(True)

    label = torch.randint(low=0, high=num_classes, size=(B * SEQ, ), device="cuda")

    ref_loss = ref_torch_linear_entropy(ref_input, ref_weight, ref_bias, label, reduction)
    loss = fused_torch_linear_entropy(input, weight, bias, label, reduction)

    torch.testing.assert_close(loss, ref_loss, atol=atol, rtol=rtol)

    ref_loss.backward()
    loss.backward()

    torch.testing.assert_close(input.grad, ref_input.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(weight.grad, ref_weight.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(bias.grad, ref_bias.grad, atol=atol, rtol=rtol)


if __name__ == "__main__":
    test_linear_entropy(4, 2, 64, 5, torch.float32, "mean", 1e-5, 5e-4)
