"""Run the Isolation Forest baseline on one C-MAPSS subset.

Validation only by default. The official test set is opened only with
--evaluate-test, which should be done once, after every choice is frozen.

    uv run python scripts/run_baseline.py --subset FD001
    uv run python scripts/run_baseline.py --subset FD001 --rolling-window 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinelts.experiment import ExperimentConfig, run_experiment
from sentinelts.models import IsolationForestDetector


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/raw"))
    p.add_argument("--subset", default="FD001")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--healthy-cycles", type=int, default=50)
    p.add_argument("--rolling-window", type=int, default=10)
    p.add_argument("--threshold-q", type=float, default=0.99)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--evaluate-test", action="store_true")
    p.add_argument("--out", type=Path, default=Path("artifacts/isolation_forest"))
    args = p.parse_args()

    cfg = ExperimentConfig(
        subset=args.subset,
        seed=args.seed,
        healthy_cycles=args.healthy_cycles,
        rolling_window=args.rolling_window,
        threshold_q=args.threshold_q,
    )
    result = run_experiment(
        args.root,
        lambda: IsolationForestDetector(n_estimators=args.n_estimators, seed=args.seed),
        cfg,
        evaluate_test=args.evaluate_test,
    )

    run_dir = args.out / f"{args.subset}_w{args.rolling_window}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(result.summary(), indent=2))
    result.validation.scored.to_csv(run_dir / "val_scores.csv", index=False)
    if result.test is not None:
        result.test.scored.to_csv(run_dir / "test_scores.csv", index=False)

    print(json.dumps(result.summary(), indent=2))
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
