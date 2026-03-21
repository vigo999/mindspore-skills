"""Portable static mHC reference adapted from a local research implementation.

This template keeps the manifold constraints and branch-routing contract while
avoiding project-specific kernels, einops, and systems optimizations.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils._pytree import tree_flatten, tree_unflatten

Projection = Literal["sinkhorn", "orthostochastic"]
InitMode = Literal["from_scratch", "checkpoint_retrofit"]


def sinkhorn_log(logits: Tensor, num_iters: int = 10, tau: float = 0.05) -> Tensor:
    """Project logits to a positive matrix with row/column sums near 1."""
    n = logits.shape[-1]
    z = logits / tau
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)
    u = torch.zeros(logits.shape[:-1], device=z.device, dtype=z.dtype)
    v = torch.zeros_like(u)

    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(z + u.unsqueeze(-1), dim=-2)

    return torch.exp(z + u.unsqueeze(-1) + v.unsqueeze(-2))


def zeropower_via_newtonschulz(
    x: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: tuple[float, float, float] = (3.0, -3.2, 1.2),
) -> Tensor:
    a, b, c = coeffs
    x = x / (x.norm() + eps)

    transpose = False
    if x.shape[0] > x.shape[1]:
        x = x.transpose(0, 1)
        transpose = True

    for _ in range(steps):
        a_mat = x @ x.transpose(0, 1)
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x

    if transpose:
        x = x.transpose(0, 1)

    return x


def orthostochastic_project(
    logits: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: tuple[float, float, float] = (3.0, -3.2, 1.2),
) -> Tensor:
    ortho = zeropower_via_newtonschulz(logits, steps=steps, eps=eps, coeffs=coeffs)
    return ortho.square()


def expand_streams_repeat(x: Tensor, num_streams: int) -> Tensor:
    if num_streams == 1:
        return x
    batch = x.shape[0]
    expanded = x.unsqueeze(1).expand(batch, num_streams, *x.shape[1:])
    return expanded.reshape(batch * num_streams, *x.shape[1:])


def expand_streams_identity_safe(x: Tensor, num_streams: int) -> Tensor:
    if num_streams == 1:
        return x
    batch = x.shape[0]
    expanded = x.new_zeros((batch, num_streams, *x.shape[1:]))
    # Keep the original signal only on stream 0 for checkpoint-safe retrofits.
    expanded[:, 0] = x
    return expanded.reshape(batch * num_streams, *x.shape[1:])


def reduce_streams_sum(x: Tensor, num_streams: int) -> Tensor:
    if num_streams == 1:
        return x
    if x.shape[0] % num_streams != 0:
        raise ValueError("batch dimension must be divisible by num_streams")
    batch = x.shape[0] // num_streams
    return x.reshape(batch, num_streams, *x.shape[1:]).sum(dim=1)


def _alpha_logit(alpha: float) -> float:
    alpha_clamped = max(1e-4, min(1.0 - 1e-4, alpha))
    return math.log(alpha_clamped / (1.0 - alpha_clamped))


def _first_tensor_index(leaves: list[Any]) -> int:
    for index, leaf in enumerate(leaves):
        if isinstance(leaf, torch.Tensor):
            return index
    raise TypeError("branch output must contain at least one tensor leaf")


class StaticMHCAdapter(nn.Module):
    """Static per-layer mHC adapter for sequence-last decoder blocks."""

    def __init__(
        self,
        num_streams: int,
        dim: int,
        *,
        branch: Callable[..., Any] | None = None,
        layer_index: int | None = None,
        init_mode: InitMode = "from_scratch",
        dropout: float = 0.0,
        residual_transform: nn.Module | None = None,
        mhc_sinkhorn_iters: int = 10,
        mhc_sinkhorn_tau: float = 0.05,
        mhc_projection: Projection = "sinkhorn",
        mhc_residual_identity_mix: bool = False,
        mhc_residual_alpha: float = 0.01,
    ) -> None:
        super().__init__()
        if num_streams <= 0:
            raise ValueError("num_streams must be positive")
        if mhc_projection not in ("sinkhorn", "orthostochastic"):
            raise ValueError("mhc_projection must be 'sinkhorn' or 'orthostochastic'")
        if init_mode not in ("from_scratch", "checkpoint_retrofit"):
            raise ValueError("unsupported init_mode")

        self.num_streams = num_streams
        self.dim = dim
        self.branch = branch
        self.init_mode = init_mode
        self.mhc_sinkhorn_iters = mhc_sinkhorn_iters
        self.mhc_sinkhorn_tau = mhc_sinkhorn_tau
        self.mhc_projection = mhc_projection
        self.mhc_residual_identity_mix = mhc_residual_identity_mix
        self.dropout = nn.Dropout(dropout)
        self.residual_transform = residual_transform or nn.Identity()

        primary_stream = 0 if init_mode == "checkpoint_retrofit" else (
            0 if layer_index is None else layer_index % num_streams
        )
        off_diag_bias = -12.0 if init_mode == "checkpoint_retrofit" else -8.0

        h_res_init = torch.full((num_streams, num_streams), off_diag_bias)
        h_res_init.fill_diagonal_(0.0)
        self.H_res_logits = nn.Parameter(h_res_init)

        h_pre_init = torch.full((num_streams,), off_diag_bias)
        h_pre_init[primary_stream] = 0.0
        self.H_pre_logits = nn.Parameter(h_pre_init)

        h_post_init = torch.full((num_streams,), off_diag_bias)
        if init_mode == "from_scratch":
            h_post_init.zero_()
        else:
            h_post_init[primary_stream] = 0.0
        self.H_post_logits = nn.Parameter(h_post_init)

        if mhc_residual_identity_mix:
            self.H_res_alpha_logit = nn.Parameter(
                torch.tensor(_alpha_logit(mhc_residual_alpha))
            )
        else:
            self.register_parameter("H_res_alpha_logit", None)

    def _batch_size(self, residuals: Tensor) -> int:
        if residuals.shape[-1] != self.dim:
            raise ValueError(f"expected hidden dim {self.dim}, got {residuals.shape[-1]}")
        if residuals.shape[0] % self.num_streams != 0:
            raise ValueError("batch dimension must be divisible by num_streams")
        return residuals.shape[0] // self.num_streams

    def _reshape_to_streams(self, residuals: Tensor) -> Tensor:
        batch = self._batch_size(residuals)
        return residuals.reshape(batch, self.num_streams, *residuals.shape[1:])

    def _reshape_from_streams(self, residuals: Tensor) -> Tensor:
        batch = residuals.shape[0]
        return residuals.reshape(batch * self.num_streams, *residuals.shape[2:])

    def project_h_res(self) -> Tensor:
        if self.mhc_projection == "orthostochastic":
            projected = orthostochastic_project(self.H_res_logits)
        else:
            projected = sinkhorn_log(
                self.H_res_logits,
                num_iters=self.mhc_sinkhorn_iters,
                tau=self.mhc_sinkhorn_tau,
            )

        if self.H_res_alpha_logit is not None:
            alpha = torch.sigmoid(self.H_res_alpha_logit)
            identity = torch.eye(
                self.num_streams,
                device=projected.device,
                dtype=projected.dtype,
            )
            projected = (1.0 - alpha) * identity + alpha * projected

        return projected

    def project_h_pre(self) -> Tensor:
        return F.softmax(self.H_pre_logits, dim=-1)

    def project_h_post(self) -> Tensor:
        return F.softmax(self.H_post_logits, dim=-1)

    def width_connection(self, residuals: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        residuals_source = self.residual_transform(residuals)
        residuals_streams = self._reshape_to_streams(residuals)
        source_streams = self._reshape_to_streams(residuals_source)

        h_res = self.project_h_res()
        h_pre = self.project_h_pre()
        h_post = self.project_h_post()

        residuals_mixed = torch.einsum("st,bs...d->bt...d", h_res, source_streams)
        branch_input = torch.einsum("s,bs...d->b...d", h_pre, residuals_streams)

        return branch_input, residuals_mixed, {"beta": h_post}

    def depth_connection(self, branch_output: Tensor, residuals: Tensor, *, beta: Tensor) -> Tensor:
        branch_to_streams = torch.einsum("s,b...d->bs...d", beta, branch_output)
        output = residuals + branch_to_streams
        output = self.dropout(output)
        return self._reshape_from_streams(output)

    def _merge_branch_output(self, branch_output: Any, residuals: Tensor, *, beta: Tensor) -> Any:
        leaves, spec = tree_flatten(branch_output)
        hidden_index = _first_tensor_index(leaves)
        leaves[hidden_index] = self.depth_connection(leaves[hidden_index], residuals, beta=beta)
        return tree_unflatten(leaves, spec)

    def forward(self, residuals: Tensor, *branch_args: Any, **branch_kwargs: Any) -> Any:
        branch_input, residuals_mixed, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_output: Any) -> Any:
            return self._merge_branch_output(branch_output, residuals_mixed, **residual_kwargs)

        if self.branch is None:
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)


__all__ = [
    "StaticMHCAdapter",
    "expand_streams_identity_safe",
    "expand_streams_repeat",
    "orthostochastic_project",
    "reduce_streams_sum",
    "sinkhorn_log",
]

