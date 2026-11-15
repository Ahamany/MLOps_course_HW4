import pytest
import torch

from ml_pipeline.models import TabularClassifier

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("batch_size", [1, 7])
def test_classifier_returns_logits_and_supports_backward(batch_size):
    model = TabularClassifier(num_features=4, embed_dim=8, num_heads=2, num_classes=3)
    logits = model(torch.randn(batch_size, 4))
    assert logits.shape == (batch_size, 3)
    loss = torch.nn.functional.cross_entropy(logits, torch.arange(batch_size) % 3)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("kwargs", [{"num_features": 0}, {"num_features": 4, "num_classes": 1}])
def test_classifier_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        TabularClassifier(**kwargs)


def test_classifier_rejects_wrong_feature_count():
    with pytest.raises(ValueError, match="features"):
        TabularClassifier(num_features=4)(torch.zeros(3, 5))
