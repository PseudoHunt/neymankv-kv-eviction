"""RoPE in the HF (rotate_half) convention, and its inverse."""
import torch


def inv_freq(head_dim: int, theta: float, device=None) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64, device=device) / head_dim))


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def cos_sin(positions: torch.Tensor, head_dim: int, theta: float, hf_fp32: bool = True):
    """hf_fp32=True reproduces transformers' default rotary embedding (angles formed in fp32),
    so inverting uses exactly the cos/sin the model used."""
    if hf_fp32:
        invf = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=positions.device).float() / head_dim))
        f = (invf[:, None].float() @ positions[None].float()).T
    else:
        f = torch.outer(positions.to(torch.float64), inv_freq(head_dim, theta, positions.device))
    emb = torch.cat((f, f), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float, inverse: bool = False,
               hf_fp32: bool = True) -> torch.Tensor:
    """x: [..., T, D]; positions: [T]. inverse=True rotates by -p. Computed in fp64, returned in x.dtype."""
    cos, sin = cos_sin(positions, x.shape[-1], theta, hf_fp32)
    cos, sin = cos.double(), sin.double()
    if inverse:
        sin = -sin
    xd = x.to(torch.float64)
    out = xd * cos + rotate_half(xd) * sin
    return out.to(x.dtype)


def apply_rope_like_hf(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Forward rotation with the same dtype path as HF (cos/sin cast to x.dtype, ops in x.dtype)."""
    cos, sin = cos_sin(positions, x.shape[-1], theta, True)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return (x * cos) + (rotate_half(x) * sin)
