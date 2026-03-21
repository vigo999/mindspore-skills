from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from portable_static_mhc import (  # noqa: E402
    StaticMHCAdapter,
    expand_streams_identity_safe,
    expand_streams_repeat,
    reduce_streams_sum,
)


def test_sinkhorn_h_res_constraints() -> None:
    adapter = StaticMHCAdapter(num_streams=4, dim=16)
    h_res = adapter.project_h_res()

    assert h_res.min().item() >= 0.0
    assert torch.allclose(h_res.sum(dim=-1), torch.ones(4), atol=1e-3)
    assert torch.allclose(h_res.sum(dim=-2), torch.ones(4), atol=1e-3)


def test_orthostochastic_h_res_constraints() -> None:
    adapter = StaticMHCAdapter(
        num_streams=4,
        dim=16,
        mhc_projection="orthostochastic",
    )
    h_res = adapter.project_h_res()

    assert h_res.min().item() >= 0.0
    assert torch.allclose(h_res.sum(dim=-1), torch.ones(4), atol=5e-2)
    assert torch.allclose(h_res.sum(dim=-2), torch.ones(4), atol=5e-2)


def test_h_pre_h_post_non_negative() -> None:
    adapter = StaticMHCAdapter(num_streams=4, dim=16)
    h_pre = adapter.project_h_pre()
    h_post = adapter.project_h_post()

    assert h_pre.min().item() >= 0.0
    assert h_post.min().item() >= 0.0
    assert torch.allclose(h_pre.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(h_post.sum(), torch.tensor(1.0), atol=1e-6)


def test_forward_shapes() -> None:
    adapter = StaticMHCAdapter(num_streams=4, dim=8)
    x = torch.randn(8, 6, 8)

    branch_input, add_residual = adapter(x)
    assert branch_input.shape == (2, 6, 8)

    out = add_residual(torch.randn(2, 6, 8))
    assert out.shape == x.shape


def test_gradient_flow() -> None:
    adapter = StaticMHCAdapter(
        num_streams=4,
        dim=8,
        mhc_residual_identity_mix=True,
    )
    x = torch.randn(8, 6, 8, requires_grad=True)

    branch_input, add_residual = adapter(x)
    out = add_residual(branch_input)
    out.sum().backward()

    assert x.grad is not None
    assert adapter.H_res_logits.grad is not None
    assert adapter.H_pre_logits.grad is not None
    assert adapter.H_post_logits.grad is not None
    assert adapter.H_res_alpha_logit is not None
    assert adapter.H_res_alpha_logit.grad is not None


def test_tuple_passthrough() -> None:
    class TupleBranch(nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            aux = x.mean(dim=-1)
            return x * 2.0, aux

    adapter = StaticMHCAdapter(num_streams=4, dim=8, branch=TupleBranch())
    x = torch.randn(8, 5, 8)

    output, aux = adapter(x)
    assert output.shape == x.shape
    assert aux.shape == (2, 5)


def test_num_streams_one_matches_residual() -> None:
    torch.manual_seed(0)
    branch = nn.Linear(8, 8)
    adapter = StaticMHCAdapter(num_streams=1, dim=8, branch=branch)
    x = torch.randn(2, 4, 8)

    expected = branch(x) + x
    actual = adapter(x)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_repeat_sum_scales_by_num_streams() -> None:
    x = torch.randn(2, 3, 8)
    out = reduce_streams_sum(expand_streams_repeat(x, 4), 4)

    torch.testing.assert_close(out, x * 4.0, atol=1e-6, rtol=1e-6)


def test_identity_safe_sum_preserves_baseline() -> None:
    x = torch.randn(2, 3, 8)
    out = reduce_streams_sum(expand_streams_identity_safe(x, 4), 4)

    torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)

