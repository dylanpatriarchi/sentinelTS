"""Run one detector on one C-MAPSS subset following docs/evaluation_protocol.md.

Validation only by default. The official test set is opened only with
--evaluate-test, which should be done once, after every choice is frozen.

    uv run python scripts/run_experiment.py --model iforest --subset FD001
    uv run python scripts/run_experiment.py --model lstm_ae --subset FD001 --rolling-window 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinelts.experiment import ExperimentConfig, run_experiment
from sentinelts.models import IsolationForestDetector, LSTMAutoencoderDetector, TrainingConfig


def build_factory(args: argparse.Namespace):
    if args.model == "iforest":
        return lambda: IsolationForestDetector(n_estimators=args.n_estimators, seed=args.seed)
    if args.model == "lstm_ae":
        train_cfg = TrainingConfig(
            epochs=args.epochs, lr=args.lr, patience=args.patience, device=args.device
        )
        return lambda: LSTMAutoencoderDetector(
            window=args.window,
            hidden_size=args.hidden_size,
            latent_size=args.latent_size,
            score_on=args.score_on,
            training=train_cfg,
            seed=args.seed,
        )
    raise ValueError(f"Unknown model {args.model}")


def run_name(args: argparse.Namespace) -> str:
    name = f"{args.subset}_w{args.rolling_window}_s{args.seed}"
    if args.model == "lstm_ae":
        name += f"_W{args.window}_h{args.hidden_size}_z{args.latent_size}_{args.score_on}"
    return name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["iforest", "lstm_ae"], default="iforest")
    p.add_argument("--root", type=Path, default=Path("data/raw"))
    p.add_argument("--subset", default="FD001")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--healthy-cycles", type=int, default=50)
    p.add_argument("--rolling-window", type=int, default=10)
    p.add_argument("--threshold-q", type=float, default=0.99)
    p.add_argument("--evaluate-test", action="store_true")
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    # Isolation Forest
    p.add_argument("--n-estimators", type=int, default=200)
    # LSTM-AE
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--latent-size", type=int, default=8)
    p.add_argument("--score-on", choices=["window", "last"], default="window")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = ExperimentConfig(
        subset=args.subset,
        seed=args.seed,
        healthy_cycles=args.healthy_cycles,
        rolling_window=args.rolling_window,
        threshold_q=args.threshold_q,
    )
    result = run_experiment(args.root, build_factory(args), cfg, evaluate_test=args.evaluate_test)

    run_dir = args.out / args.model / run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = result.summary() | {
        "model": args.model,
        "cli_args": vars(args) | {"root": str(args.root), "out": str(args.out)},
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    result.validation.scored.to_csv(run_dir / "val_scores.csv", index=False)
    if result.test is not None:
        result.test.scored.to_csv(run_dir / "test_scores.csv", index=False)

    printable = {k: v for k, v in summary.items() if k != "training_history"}
    print(json.dumps(printable, indent=2))
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
