"""Attention-output estimators with logit bias, and the error metrics."""
import torch


def logits(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return (Q.float() @ K.float().T) / (K.shape[-1] ** 0.5)


def full_output(L: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return torch.softmax(L, dim=-1) @ V.float()


def est_output(L: torch.Tensor, V: torch.Tensor, idx: torch.Tensor, bias: torch.Tensor | None = None):
    Ls = L[:, idx]
    if bias is not None:
        Ls = Ls + bias.to(L)
    return torch.softmax(Ls, dim=-1) @ V.float()[idx]


def rel_err(o_hat: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """Per-query ||o_hat - o||^2 / ||o||^2."""
    return ((o_hat - o) ** 2).sum(-1) / (o ** 2).sum(-1).clamp_min(1e-30)


def mass_recall(P: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Per-query fraction of full attention mass on kept tokens."""
    return P[:, idx].sum(-1)


def summarize(err: torch.Tensor) -> tuple[float, float]:
    return float(err.mean()), float(torch.quantile(err.double(), 0.95))
