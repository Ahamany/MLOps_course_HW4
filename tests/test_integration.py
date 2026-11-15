from dataclasses import asdict

import pytest
import torch

from ml_pipeline.data import TableDataset, make_classification
from ml_pipeline.training import (
    TrainConfig,
    evaluate,
    load_checkpoint,
    predict,
    run_training,
    save_checkpoint,
)

pytestmark = pytest.mark.integration


def test_real_pipeline_learns_and_is_reproducible():
    data = make_classification(256, seed=42)
    config = TrainConfig(epochs=20, seed=42)
    result = run_training(data, config)
    assert result.history[-1]["valid_loss"] < result.initial_loss * 0.6
    assert result.history[-1]["valid_accuracy"] >= 0.85
    assert evaluate(result, make_classification(128, seed=987))["accuracy"] >= 0.85
    again = run_training(data, config)
    assert result.history == again.history
    for name, value in result.model.state_dict().items():
        torch.testing.assert_close(value, again.model.state_dict()[name], atol=0, rtol=0)


def test_checkpoint_restores_model_scaler_config_and_predictions(tmp_path):
    data = make_classification(64)
    result = run_training(data, TrainConfig(epochs=3))
    path = tmp_path / "nested" / "model.pt"
    save_checkpoint(result, path)
    loaded = load_checkpoint(path)
    assert asdict(loaded.config) == asdict(result.config)
    assert loaded.history == result.history and loaded.initial_loss == result.initial_loss
    torch.testing.assert_close(loaded.scaler.mean, result.scaler.mean)
    torch.testing.assert_close(loaded.scaler.scale, result.scaler.scale)
    assert torch.equal(predict(loaded, data.features), predict(result, data.features))
    assert evaluate(loaded, data) == evaluate(result, data)
    with pytest.raises(ValueError, match="classes"):
        run_training(TableDataset([[1, 2], [3, 4]], [0, 2]), TrainConfig(epochs=1))
    torch.save({"format_version": 999}, path)
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(path)
