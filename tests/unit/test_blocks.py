import pytest
import torch

from ml_pipeline.blocks import AttentionBlock, LogAttention

pytestmark = pytest.mark.unit


def test_attention_matches_uniform_average_with_identity_value_projection():
    attention = LogAttention(embed_dim=4, num_heads=2, bias=False)
    with torch.no_grad():
        attention.q_proj.weight.zero_()
        attention.k_proj.weight.zero_()
        attention.v_proj.weight.copy_(torch.eye(4))
        attention.out_proj.weight.copy_(torch.eye(4))
    query = torch.randn(2, 3, 4)
    value = torch.randn(2, 5, 4)
    output, weights = attention(query, value, value, need_weights=True)
    torch.testing.assert_close(output, value.mean(dim=1, keepdim=True).expand(2, 3, 4))
    torch.testing.assert_close(weights, torch.full((2, 3, 5), 0.2))


def test_residual_block_backpropagates_through_every_parameter():
    block = AttentionBlock(embed_dim=8, num_heads=2)
    x = torch.randn(3, 4, 8, requires_grad=True)
    output = block(x)
    assert output.shape == x.shape
    output.square().mean().backward()
    for parameter in [x, *block.parameters()]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    "kwargs",
    [{"embed_dim": 0, "num_heads": 2}, {"embed_dim": 5, "num_heads": 2},
     {"embed_dim": 4, "num_heads": 0}, {"embed_dim": 4, "num_heads": 2, "dropout": -0.1},
     {"embed_dim": 4, "num_heads": 2, "dropout": 1.0}],
)
def test_attention_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        LogAttention(**kwargs)


@pytest.mark.parametrize("case", ["rank", "embedding", "batch", "length", "padding", "mask"])
def test_attention_rejects_invalid_shapes(case):
    attention = LogAttention(4, 2)
    q, k, v = (torch.randn(2, 3, 4) for _ in range(3))
    kwargs = {}
    if case == "rank":
        q = q[0]
    elif case == "embedding":
        k = torch.randn(2, 3, 5)
    elif case == "batch":
        k = torch.randn(1, 3, 4)
    elif case == "length":
        v = torch.randn(2, 2, 4)
    elif case == "padding":
        kwargs["key_padding_mask"] = torch.zeros(2, 2, dtype=torch.bool)
    else:
        kwargs["attn_mask"] = torch.zeros(3, 3)  # Boolean masks only.
    with pytest.raises(ValueError):
        attention(q, k, v, **kwargs)
