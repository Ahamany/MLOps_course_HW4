import pytest
import torch

from ml_pipeline.blocks import LogAttention

pytestmark = pytest.mark.regression


def test_negative_projections_and_fully_masked_rows_have_finite_gradients():
    attention = LogAttention(4, 2)
    with torch.no_grad():
        attention.q_proj.weight.fill_(-1)
        attention.k_proj.weight.fill_(-1)
        attention.q_proj.bias.fill_(-10)
        attention.k_proj.bias.fill_(-10)
    x = torch.ones(2, 3, 4, requires_grad=True)
    padding = torch.tensor([[True, True, True], [False, False, True]])
    output, weights = attention(x, x, x, key_padding_mask=padding, need_weights=True)
    assert torch.isfinite(output).all() and torch.isfinite(weights).all()
    assert torch.count_nonzero(weights[0]) == 0
    assert torch.count_nonzero(weights[:, :, 2]) == 0
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in attention.parameters())


def test_causal_and_explicit_masks_apply_even_when_returning_weights():
    attention = LogAttention(4, 2, dropout=0.5).eval()
    x = torch.randn(2, 4, 4)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[:, 0] = True
    before = mask.clone()
    output, weights = attention(x, x, x, attn_mask=mask, is_causal=True, need_weights=True)
    without_weights, absent = attention(x, x, x, attn_mask=mask, is_causal=True)
    torch.testing.assert_close(output, without_weights, atol=0, rtol=0)
    assert absent is None
    assert torch.equal(mask, before)
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0
    assert torch.count_nonzero(weights[:, :, 0]) == 0
    torch.testing.assert_close(weights[:, 1:].sum(-1), torch.ones(2, 3))
    changed = x.clone()
    changed[:, -1] += 100
    altered, _ = attention(changed, changed, changed, is_causal=True)
    original, _ = attention(x, x, x, is_causal=True)
    torch.testing.assert_close(altered[:, :-1], original[:, :-1])
