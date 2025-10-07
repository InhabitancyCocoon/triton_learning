"""

TODO:

cutlass, cute
fp8, wait, it doesn't seem rational to compute cross entropy with fp8 precision.

ref link:

https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py

https://github.com/mgmalek/efficient_cross_entropy/blob/main/test.py

https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

https://discuss.pytorch.org/t/the-difference-between-torch-tensor-data-and-torch-tensor/25995

https://docs.pytorch.org/docs/stable/generated/torch.gather.html

https://www.cnblogs.com/zzk0/p/15173022.html

https://docs.pytorch.org/docs/stable/amp.html

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

        # linear y = x @ weight.T + bias
        logit = input @ weight.T + bias[None, :]  # N x C
        logit_row_max = logit.max(dim=1, keepdim=True).values

        # stable row-wise softmax
        logit = logit - logit_row_max
        logit_exp = logit.exp()
        logit_exp_row_sum = logit_exp.sum(dim=1, keepdim=True)
        logit_softmax = logit_exp / logit_exp_row_sum

        # log
        logit_log_softmax = logit_softmax.log()
        cross_entropy = -logit_log_softmax.gather(dim=1, index=label[:, None]).squeeze(1)

        ctx.num_tokens = input.shape[0]
        ctx.H = input.shape[1]
        ctx.vocab_size = weight.shape[0]
        ctx.reduction = reduction

        ctx.save_for_backward(
            input,
            weight,
            label,
            logit_softmax
        )  # should be called only once

        if reduction == 'mean':
            return cross_entropy.mean()
        elif reduction == 'sum':
            return cross_entropy.sum()

    @staticmethod
    def backward(ctx, grad_output):
        """
        If the forward output is a scalar, grad_output is a tensor scalar with value 1,
        else the grad_output is a tensor with same shape as the forward output.
        """
        grad_input = grad_weight = grad_bias = None
        (
            input,
            weight,
            label,
            logit_softmax,
        ) = ctx.saved_tensors

        if ctx.reduction == "mean":
            scale = 1.0 / ctx.num_tokens
            grad_reduction = torch.empty(ctx.num_tokens, device=input.device).fill_(1.0 / ctx.num_tokens)
        elif ctx.reduction == "sum":
            scale = 1.0
            grad_reduction = torch.ones(ctx.num_tokens, device=input.device)

        # Optimization 1: compute the grad_softmax in-place, some smart math happens here.

        grad_softmax = (scale * logit_softmax).scatter_add_(dim=1, index=label[:, None], src=-grad_reduction[:, None])

        # linear backward, use the kernel in /extra/linear.py


        grad_input = grad_softmax @ weight
        grad_weight = grad_softmax.T @ input
        grad_bias = grad_softmax.sum(dim=0)

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
    output = F.linear(input, weight, bias)
    entropy = F.cross_entropy(output, label, reduction=reduction)
    return entropy


# pytest.mark.parametrize is unhappy with default param.
@pytest.mark.parametrize(
    "B, SEQ, H, num_classes, reduction",
    itertools.product(
        [1, 8, 17, 33],
        [1, 4, 7, 11, 256, 257],
        [1, 3, 7, 16],
        [1, 255, 512, 1023],
        ["mean", "sum"],  # we assume the output of fused linear entropy is a scalar.
    )
)
def test_linear_entropy(B, SEQ, H, num_classes, reduction):
    torch.manual_seed(43)

    ref_input = torch.rand(B * SEQ, H, requires_grad=True, device="cuda")
    ref_weight = torch.empty(num_classes, H, requires_grad=True, device="cuda")
    # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
    with torch.no_grad():
        ref_weight.uniform_()
    ref_bias = torch.rand(num_classes, requires_grad=True, device="cuda")

    input = ref_input.detach().clone().requires_grad_(True)
    weight = ref_weight.detach().clone().requires_grad_(True)
    bias = ref_bias.detach().clone().requires_grad_(True)

    label = torch.randint(low=0, high=num_classes, size=(B * SEQ, ), device="cuda")

    ref_loss = ref_torch_linear_entropy(ref_input, ref_weight, ref_bias, label, reduction)
    loss = fused_torch_linear_entropy(input, weight, bias, label, reduction)

    torch.testing.assert_close(loss, ref_loss, atol=1e-4, rtol=1e-4)

    ref_loss.backward()
    loss.backward()

    torch.testing.assert_close(ref_input.grad, input.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(ref_weight.grad, weight.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(ref_bias.grad, bias.grad, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_linear_entropy(3, 16, 32, 7, "mean")
