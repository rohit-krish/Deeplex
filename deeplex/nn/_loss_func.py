from ..engine import Tensor


def _check_if_same_device(y1: Tensor, y2: Tensor):
    if y1.device != y2.device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but found at least two devices, cuda and cpu!"
        )


def MSELoss(y1: Tensor, y2: Tensor) -> Tensor:
    """
    Compute the Mean Squared Error (MSE) loss between two tensors.

    Args:
        y1 (Tensor): The first input tensor.
        y2 (Tensor): The second input tensor.

    Returns:
        Tensor: The computed MSE loss.
    """

    _check_if_same_device(y1, y2)
    return ((y1 - y2) ** 2).sum() / len(y2)


def BCELoss(input: Tensor, pred: Tensor, eps=1e-7) -> Tensor:
    """
    Compute the Binary Cross Entropy (BCE) loss between predicted and target tensors.

    Args:
        input (Tensor): The target tensor.
        pred (Tensor): The predicted tensor.
        eps (float, optional): A small constant to avoid numerical instability. Default is 1e-7.

    Returns:
        Tensor: The computed BCE loss.
    """

    _check_if_same_device(input, pred)

    # clipping to avoid numerical instability
    a = pred * input.clip(eps, 1 - eps).log()
    b = (1 - pred) * (1 - input).clip(eps, 1 - eps).log()
    return -(a + b).sum() / len(pred)


def NLLLoss(input: Tensor, pred: Tensor, eps=1e-7) -> Tensor:
    """
    Compute the Negative Log-Likelihood (NLL) loss between predicted and target tensors.

    Args:
        input (Tensor): The target tensor.
        pred (Tensor): The predicted tensor.
        eps (float, optional): A small constant to avoid numerical instability. Default is 1e-7.

    Returns:
        Tensor: The computed NLL loss.

    """

    _check_if_same_device(input, pred)

    # clipping to avoid numerical instability
    return -(input * pred.clip(eps, 1 - eps).log()).sum() / len(pred)
