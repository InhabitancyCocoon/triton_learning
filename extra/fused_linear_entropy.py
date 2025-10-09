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

            chunk_input = input[start_token_idx : end_token_idx]
            chunk_label = label[start_token_idx : end_token_idx]

            chunk_logit = chunk_input @ weight.T + bias[None, :]

            # the below operations can be fused into a triton kernel
            # 1. online softmax happens here (fp32 precision)
            # 2. compute the cross entropy for each token
            # 3. update the logit in-place (scatter_add_ and mul)
            

            chunk_logit_row_max = chunk_logit.float().max(dim=1, keepdim=True).values
            chunk_logit = chunk_logit - chunk_logit_row_max
            chunk_logit_exp = chunk_logit.exp()
            chunk_logit_exp_row_sum = chunk_logit_exp.sum(dim=1, keepdim=True)
            chunk_logit_softmax = chunk_logit_exp / chunk_logit_exp_row_sum

            chunk_cross_entropy = - chunk_logit_softmax.gather(dim=1,index=chunk_label[:, None]) \
                                                       .squeeze(1).log()

            loss_per_token[start_token_idx : end_token_idx] = chunk_cross_entropy


            chunk_logit_softmax.scatter_add_(
                dim=1, index=chunk_label[:, None], src=-torch.ones_like(chunk_logit_softmax)
            )

            chunk_grad_softmax = chunk_logit_softmax * scale

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
    reduction: str
):
    output = F.linear(input, weight, bias).float()
    entropy = F.cross_entropy(output, label, reduction=reduction)
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
        (torch.float32, "mean", 1e-5, 5e-4),
        (torch.bfloat16, "mean", 5e-3, 5e-2),
        (torch.float32, "sum", 1e-3, 5e-2),
        (torch.bfloat16, "sum", 5e0, 5e-1),
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

    torch.testing.assert_close(ref_input.grad, input.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(ref_weight.grad, weight.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(ref_bias.grad, bias.grad, atol=atol, rtol=rtol)


if __name__ == "__main__":
    test_linear_entropy(32, 512, 64, 1936, "mean")
