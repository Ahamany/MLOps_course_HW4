"""Command-line entry point. Successful commands print one JSON object."""

import argparse
import json
from pathlib import Path

import torch

from ml_pipeline.data import TableDataset, make_classification
from ml_pipeline.training import TrainConfig, evaluate, load_checkpoint, run_training, save_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU tabular classification pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train and save checkpoint and JSON metrics")
    train.add_argument("--output", type=Path, default=Path("artifacts/run"))
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=0.01)
    train.add_argument("--features", type=int, default=4)
    assess = commands.add_parser("evaluate", help="evaluate a saved model on another dataset")
    assess.add_argument("--checkpoint", type=Path, required=True)
    for command in (train, assess):
        command.add_argument("--csv", type=Path)
        command.add_argument("--target", default="target")
        command.add_argument("--samples", type=int, default=256)
        command.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        if args.command == "train":
            config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                                 learning_rate=args.learning_rate, seed=args.seed)
            dataset = (TableDataset.from_csv(args.csv, args.target) if args.csv else
                       make_classification(args.samples, args.features, args.seed))
            result = run_training(dataset, config)
            checkpoint_path = args.output / "model.pt"
            save_checkpoint(result, checkpoint_path)
            metrics = {
                "checkpoint": str(checkpoint_path), "initial_loss": result.initial_loss,
                "final_loss": result.history[-1]["valid_loss"],
                "accuracy": result.history[-1]["valid_accuracy"],
            }
            (args.output / "history.json").write_text(
                json.dumps(result.history, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            (args.output / "metrics.json").write_text(
                json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        else:
            result = load_checkpoint(args.checkpoint)
            dataset = (TableDataset.from_csv(args.csv, args.target) if args.csv else
                       make_classification(args.samples, result.model.num_features, args.seed))
            metrics = evaluate(result, dataset)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(metrics, allow_nan=False))
    return 0
