"""CLI entrypoint for mandible-rvc."""
from __future__ import annotations

import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mandible_rvc.trainer import run_training
else:
    from .trainer import run_training


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="mandible-rvc",
        description="Headless RVC v2 voice model trainer and ONNX exporter for Mandible.",
    )
    sub = parser.add_subparsers(dest="command")

    train_parser = sub.add_parser("train", help="Train a custom voice model from vocal recordings")
    train_parser.add_argument("--input", "-i", required=True, nargs="+", help="Path(s) to vocal audio file(s) or directories")
    train_parser.add_argument("--name", "-n", required=True, help="Display name of the voice model")
    train_parser.add_argument("--output-dir", "-o", required=True, help="Directory to save the resulting .onnx and .index files")
    train_parser.add_argument("--epochs", "-e", type=int, default=200, help="Number of training epochs (default: 200)")
    train_parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size (default: 8)")
    train_parser.add_argument("--f0-method", default="hybrid", help="Pitch tracking algorithm ('hybrid', 'rmvpe', 'harvest', 'crepe', 'pm')")
    train_parser.add_argument("--sample-rate", "-sr", type=int, default=40000, help="Model sample rate (e.g. 40000, 44100, 48000)")
    train_parser.add_argument("--device", default="auto", help="Training accelerator device ('auto', 'cuda:0', 'dml', 'cpu')")

    args = parser.parse_args()
    if args.command == "train":
        return run_training(
            input_paths=args.input,
            voice_name=args.name,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            f0_method=args.f0_method,
            sample_rate=args.sample_rate,
            device=args.device,
        )

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
