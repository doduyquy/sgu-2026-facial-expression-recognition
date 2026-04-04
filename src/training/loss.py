# mypy: allow-untyped-defs
from collections.abc import Callable
from typing_extensions import deprecated

from torch import Tensor
from torch.nn import _reduction as _Reduction, functional as F

from torch import Module


__all__ = [
    "FocalLoss",
]


class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.weight: Tensor | None



class FocalLoss(_WeightedLoss):
    r"""This criterion computes the focal loss between input logits and target.

    Focal loss is an extension of cross entropy loss that down-weights easy examples
    and focuses training on hard examples. It is particularly useful for classification
    tasks with class imbalance.

    The input is expected to contain unnormalized logits for each class.
    The target should contain class indices in the range :math:`[0, C)`.

    The unreduced loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - \alpha_{y_n} (1 - p_{t,n})^\gamma \log(p_{t,n})
        \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}

    where :math:`p_{t,n}` is the predicted probability for the true class of sample :math:`n`,
    :math:`\alpha` is the optional class weight, and :math:`\gamma` is the focusing parameter.

    If :attr:`reduction` is not ``'none'`` (default ``'mean'``), then:

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, it has to be a Tensor of size `C`.
        gamma (float, optional): focusing parameter :math:`\gamma \geq 0`.
            Default: ``2.0``.
        size_average (bool, optional): Deprecated (see :attr:`reduction`).
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default: ``-100``.
        reduce (bool, optional): Deprecated (see :attr:`reduction`).
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``.

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`.
        - Target: :math:`(N)` where each value is :math:`0 \leq targets[i] < C`.
        - Output: If :attr:`reduction` is ``'none'``, then :math:`(N)`.
          Otherwise, scalar.

    Examples::

        >>> loss = nn.FocalLoss(gamma=2.0)
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    __constants__ = ["ignore_index", "reduction", "gamma"]
    ignore_index: int
    gamma: float

    def __init__(
        self,
        weight: Tensor | None = None,
        gamma: float = 2.0,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        if gamma < 0:
            raise ValueError(
                f"FocalLoss: expected gamma to be >= 0, got {gamma} instead"
            )
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.ndim != 2:
            raise ValueError(
                f"FocalLoss: expected input to have shape (N, C), but got shape {tuple(input.shape)}"
            )
        if target.ndim != 1:
            raise ValueError(
                f"FocalLoss: expected target to have shape (N,), but got shape {tuple(target.shape)}"
            )
        if input.size(0) != target.size(0):
            raise ValueError(
                f"FocalLoss: expected input batch size ({input.size(0)}) "
                f"to match target batch size ({target.size(0)})"
            )

        log_probs = F.log_softmax(input, dim=1)
        valid_mask = target != self.ignore_index

        if not valid_mask.any():
            if self.reduction == "none":
                return torch.zeros_like(target, dtype=input.dtype)
            return input.new_tensor(0.0)

        filtered_log_probs = log_probs[valid_mask]
        filtered_target = target[valid_mask]

        log_pt = filtered_log_probs.gather(1, filtered_target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        focal_factor = (1 - pt) ** self.gamma
        loss = -focal_factor * log_pt

        if self.weight is not None:
            alpha_t = self.weight.gather(0, filtered_target)
            loss = alpha_t * loss

        if self.reduction == "none":
            output = torch.zeros_like(target, dtype=input.dtype)
            output[valid_mask] = loss
            return output
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

