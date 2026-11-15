import csv
import json
import os
import subprocess
import sys

import pytest

from ml_pipeline.data import make_classification

pytestmark = pytest.mark.e2e


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "ml_pipeline", *map(str, args)],
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )


def test_cli_trains_from_csv_saves_artifacts_and_evaluates_in_fresh_process(tmp_path):
    dataset = make_classification(256)
    csv_path = tmp_path / "input.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["a", "b", "c", "d", "target"])
        for features, label in dataset:
            writer.writerow([*features.tolist(), label.item()])
    output = tmp_path / "run"
    trained = run_cli("train", "--csv", csv_path, "--output", output, "--epochs", 20)
    assert trained.returncode == 0, trained.stderr
    summary = json.loads(trained.stdout)
    assert (output / "model.pt").is_file()
    assert len(json.loads((output / "history.json").read_text())) == 20
    assert json.loads((output / "metrics.json").read_text()) == summary
    assert summary["accuracy"] >= 0.85
    assert summary["final_loss"] < summary["initial_loss"]
    evaluated = run_cli(
        "evaluate", "--checkpoint", output / "model.pt", "--samples", 128, "--seed", 987
    )
    assert evaluated.returncode == 0, evaluated.stderr
    assert json.loads(evaluated.stdout)["accuracy"] >= 0.85


def test_cli_synthetic_training_and_invalid_input_exit_codes(tmp_path):
    trained = run_cli("train", "--samples", 16, "--epochs", 1, "--output", tmp_path / "ok")
    assert trained.returncode == 0, trained.stderr
    output = tmp_path / "bad"
    for args in [
        ("train", "--epochs", 0, "--output", output),
        ("train", "--csv", tmp_path / "absent.csv", "--output", output),
        ("evaluate", "--checkpoint", tmp_path / "missing.pt"),
    ]:
        failed = run_cli(*args)
        assert failed.returncode == 2
        assert "error:" in failed.stderr
        assert not output.exists()
