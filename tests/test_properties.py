import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from ml_pipeline.blocks import LogAttention
from ml_pipeline.data import build_loaders, make_classification

pytestmark = pytest.mark.property


@settings(max_examples=20, deadline=None, derandomize=True)
@given(batch=st.integers(1, 5), length=st.integers(1, 8), heads=st.sampled_from([1, 2, 4]))
def test_attention_preserves_batch_permutations_and_probability_normalization(batch, length, heads):
    torch.manual_seed(10)
    attention = LogAttention(8, heads).eval()
    x = torch.randn(batch, length, 8)
    permutation = torch.arange(batch - 1, -1, -1)
    output, weights = attention(x, x, x, need_weights=True)
    permuted, _ = attention(x[permutation], x[permutation], x[permutation])
    torch.testing.assert_close(permuted, output[permutation])
    torch.testing.assert_close(weights.sum(-1), torch.ones(batch, length))
    assert (weights >= 0).all()


@settings(max_examples=20, deadline=None, derandomize=True)
@given(samples=st.integers(5, 50), batch_size=st.integers(1, 16), seed=st.integers(0, 1000))
def test_loaders_cover_each_sample_once_for_arbitrary_batch_sizes(samples, batch_size, seed):
    split = build_loaders(make_classification(samples, seed=seed), batch_size=batch_size, seed=seed)
    indices = split.train_indices + split.valid_indices
    assert sorted(indices) == list(range(samples))
    assert len(set(indices)) == samples
    assert sum(len(y) for _, y in split.train) == len(split.train_indices)
    assert sum(len(y) for _, y in split.valid) == len(split.valid_indices)
    assert all(torch.isfinite(x).all() for x, _ in split.train)
