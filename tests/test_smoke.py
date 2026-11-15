import pytest

from ml_pipeline.data import make_classification
from ml_pipeline.training import TrainConfig, run_training


@pytest.mark.smoke
def test_public_api_completes_one_epoch():
    result = run_training(make_classification(16), TrainConfig(epochs=1, batch_size=4))
    assert len(result.history) == 1
    assert result.history[0]["epoch"] == 1
    assert 0 <= result.history[0]["valid_accuracy"] <= 1
