import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml_pipeline.data import make_classification
from ml_pipeline.training import (
    TrainConfig,
    evaluate_model,
    load_checkpoint,
    predict,
    run_training,
    train_epoch,
)

pytestmark = pytest.mark.unit


def test_evaluation_aggregates_by_sample_and_preserves_weights_and_mode():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    x = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
    y = torch.tensor([0, 1, 0])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    before = model.weight.detach().clone()
    metrics = evaluate_model(model, loader)
    expected_loss = torch.nn.functional.cross_entropy(model(x), y).item()
    assert metrics["loss"] == pytest.approx(expected_loss)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    torch.testing.assert_close(model.weight, before, atol=0, rtol=0)
    assert model.training
    assert model.weight.grad is None


def test_train_epoch_updates_weights_and_rejects_empty_loader():
    model = torch.nn.Linear(2, 2)
    before = model.weight.detach().clone()
    loader = DataLoader(TensorDataset(torch.eye(2), torch.tensor([0, 1])), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assert train_epoch(model, loader, optimizer) > 0
    assert not torch.equal(before, model.weight)
    empty = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="empty"):
        train_epoch(model, empty, optimizer)
    with pytest.raises(ValueError, match="empty"):
        evaluate_model(model, empty)


def test_train_epoch_rejects_nonfinite_loss():
    model = torch.nn.Linear(2, 2)
    loader = DataLoader(TensorDataset(torch.full((1, 2), float("nan")), torch.tensor([0])))
    with pytest.raises(ValueError, match="finite"):
        train_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.1))


def test_predict_rejects_nonfinite_features():
    result = run_training(make_classification(8), TrainConfig(epochs=1))
    with pytest.raises(ValueError, match="finite"):
        predict(result, [[float("inf"), 0, 0, 0]])


@pytest.mark.parametrize("content", [[], {"format_version": 1}])
def test_load_checkpoint_rejects_incomplete_payloads(tmp_path, content):
    path = tmp_path / "model.pt"
    torch.save(content, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_load_checkpoint_rejects_corrupted_file(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="checkpoint"):
        load_checkpoint(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0},
        {"learning_rate": float("nan")},
        {"seed": -1},
        {"valid_fraction": 1},
        {"embed_dim": 3},
        {"num_heads": 0},
        {"num_classes": 1},
    ],
)
def test_training_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)
