import numpy as np
import pytest
import torch

from ml_pipeline.data import Standardizer, TableDataset, build_loaders, make_classification

pytestmark = pytest.mark.unit


def test_dataset_copies_inputs_and_returns_correct_tensor_types():
    features = np.array([[1, 2], [3, 4]], dtype=np.float64)
    labels = np.array([0, 1])
    dataset = TableDataset(features, labels)
    features[:] = 99
    labels[:] = 99
    x, y = dataset[1]
    assert len(dataset) == 2
    assert x.dtype == torch.float32 and y.dtype == torch.long
    assert x.tolist() == [3.0, 4.0] and y.item() == 1
    with pytest.raises(IndexError):
        dataset[2]


@pytest.mark.parametrize(
    ("features", "labels"),
    [
        ([], []),
        ([[1, 2]], [0, 1]),
        ([[float("nan")]], [0]),
        ([[float("inf")]], [0]),
        ([[1]], [-1]),
        ([[1]], [0.5]),
        ([[1]], [float("nan")]),
        ([[1]], [1e100]),
        ([[1]], [[0]]),
        ([1, 2], [0, 1]),
    ],
)
def test_dataset_rejects_invalid_data(features, labels):
    with pytest.raises(ValueError):
        TableDataset(features, labels)


def test_split_is_disjoint_reproducible_and_scaler_uses_training_data_only():
    dataset = TableDataset([[0, 4], [1, 4], [2, 4], [100, 4], [200, 4]], [0, 1, 0, 1, 0])
    split = build_loaders(dataset, batch_size=2, valid_fraction=0.4, seed=17)
    again = build_loaders(dataset, batch_size=2, valid_fraction=0.4, seed=17)
    assert split.train_indices == again.train_indices
    assert set(split.train_indices).isdisjoint(split.valid_indices)
    assert sorted(split.train_indices + split.valid_indices) == list(range(5))
    train_x = dataset.features[split.train_indices]
    torch.testing.assert_close(split.scaler.mean, train_x.mean(0))
    torch.testing.assert_close(
        split.scaler.transform(train_x).mean(0), torch.zeros(2), atol=1e-6, rtol=0
    )
    assert split.scaler.scale[1].item() == 1.0  # Constant feature must stay finite.
    assert [y.tolist() for _, y in split.train] == [y.tolist() for _, y in again.train]
    assert sum(len(y) for _, y in split.train) == 3  # Keep the incomplete last batch.


@pytest.mark.parametrize(
    "kwargs", [{"batch_size": 0}, {"valid_fraction": 0}, {"valid_fraction": 1}, {"seed": -1}]
)
def test_loader_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        build_loaders(make_classification(10), **kwargs)


def test_csv_reads_numeric_features_in_header_order_and_reports_bad_rows(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("a,target,b\n1,0,2\n3,1,4\n", encoding="utf-8")
    dataset = TableDataset.from_csv(path)
    assert dataset.features.tolist() == [[1, 2], [3, 4]]
    path.write_text("a,target\ntext,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV"):
        TableDataset.from_csv(path)


@pytest.mark.parametrize(
    "text",
    ["a,b\n1,2\n", "target\n0\n", "a,a,target\n1,2,0\n", "a,target\n1\n", "a,target\n1,0,2\n", ""],
)
def test_csv_rejects_invalid_schema(tmp_path, text):
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        TableDataset.from_csv(path)


def test_standardizer_rejects_wrong_shape_and_loader_requires_two_samples():
    scaler = Standardizer.fit(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError):
        scaler.transform(torch.zeros(2, 3))
    with pytest.raises(ValueError):
        build_loaders(TableDataset([[1, 2]], [0]))


@pytest.mark.parametrize(
    "features", [torch.empty(0, 2), torch.zeros(2), torch.tensor([[float("nan")]])]
)
def test_standardizer_rejects_invalid_training_data(features):
    with pytest.raises(ValueError):
        Standardizer.fit(features)


@pytest.mark.parametrize("kwargs", [{"n_samples": 1}, {"n_features": 1}, {"seed": -1}])
def test_synthetic_data_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        make_classification(**kwargs)
