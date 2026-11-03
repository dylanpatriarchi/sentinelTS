"""Command-line interface.

    sentinelts download                       fetch + checksum C-MAPSS into data/raw
    sentinelts explore [--subset FD001]       dataset statistics
    sentinelts run --model lstm_ae ...        one experiment (validation only by default)
    sentinelts compare [--metric ...]         mean ± std over seeds of every saved run

The official test set is opened only by ``run --evaluate-test``; see
docs/evaluation_protocol.md for when that is allowed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from sentinelts.data.cmapss import SUBSETS
from sentinelts.models.registry import MODEL_DEFAULTS, TRAINING_DEFAULTS

DEFAULT_ROOT = Path("data/raw")
DEFAULT_ARTIFACTS = Path("artifacts")


# --------------------------------------------------------------------------- download
def cmd_download(args: argparse.Namespace) -> int:
    from sentinelts.data.download import download_cmapss

    files = download_cmapss(args.dest)
    print(f"C-MAPSS ready in {args.dest} ({len(files)} files)")
    return 0


# --------------------------------------------------------------------------- explore
def cmd_explore(args: argparse.Namespace) -> int:
    from sentinelts.data.explore import describe_subset

    for subset in args.subset or SUBSETS:
        stats = describe_subset(args.root, subset, args.anomaly_rul, args.healthy_rul)
        print(f"\n=== {subset} ===")
        for key, value in stats.items():
            if key != "subset":
                print(f"  {key:32s} {value}")
    return 0


# --------------------------------------------------------------------------- run
def cmd_run(args: argparse.Namespace) -> int:
    from sentinelts.experiment import ExperimentConfig, run_experiment
    from sentinelts.models.registry import build_factory, model_params
    from sentinelts.runs import save_run

    if args.evaluate_test:
        print(
            "WARNING: evaluating on the official test set. Per the protocol this must be "
            "done once, with configuration frozen beforehand.",
            file=sys.stderr,
        )
    cfg = ExperimentConfig(
        subset=args.subset,
        seed=args.seed,
        healthy_cycles=args.healthy_cycles,
        rolling_window=args.rolling_window,
        threshold_q=args.threshold_q,
        score_smoothing=args.score_smoothing,
    )
    params = model_params(args.model, vars(args))
    result = run_experiment(
        args.root,
        build_factory(args.model, params, args.seed),
        cfg,
        evaluate_test=args.evaluate_test,
    )
    run_dir = save_run(result, args.model, params, args.artifacts)

    for split in ("validation", "test"):
        res = getattr(result, split)
        if res is None:
            continue
        p, e = res.point, res.event
        print(
            f"[{split}] early_warning_auc={p['early_warning_auc']:.3f} "
            f"pr_auc={p['pr_auc']:.3f} f1={p['f1']:.3f} "
            f"detection={e['detection_rate']:.2f} lead={e['median_lead_time']:.1f} "
            f"false_alarm_engines={e['false_alarm_unit_rate']:.2f}"
        )
    print(f"saved {run_dir}")
    return 0


# --------------------------------------------------------------------------- compare
def cmd_compare(args: argparse.Namespace) -> int:
    from sentinelts.runs import load_runs, summarize

    runs = load_runs(args.artifacts)
    if args.subset:
        runs = runs[runs["subset"].isin(args.subset)] if not runs.empty else runs
    table = summarize(runs, args.metric, args.split)
    if table.empty:
        print(f"No {args.split} runs found under {args.artifacts}")
        return 1
    if args.json:
        print(table.to_json(orient="records", indent=2))
    else:
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(table.round(3).to_string(index=False))
    return 0


# --------------------------------------------------------------------------- parser
def _add_model_options(p: argparse.ArgumentParser) -> None:
    """One CLI option per model hyper-parameter, defaults taken from the registry."""
    seen: dict[str, object] = {}
    for params in [*MODEL_DEFAULTS.values(), TRAINING_DEFAULTS]:
        for name, default in params.items():
            seen.setdefault(name, default)
    group = p.add_argument_group("model hyper-parameters (only those of --model are used)")
    for name, default in seen.items():
        kwargs: dict = {"default": default, "type": type(default)}
        if name == "score_on":
            kwargs["choices"] = ["window", "last"]
        group.add_argument(f"--{name.replace('_', '-')}", **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentinelts", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download and checksum C-MAPSS")
    p.add_argument("--dest", type=Path, default=DEFAULT_ROOT)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("explore", help="dataset statistics")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--subset", nargs="*", choices=SUBSETS)
    p.add_argument("--anomaly-rul", type=int, default=30)
    p.add_argument("--healthy-rul", type=int, default=100)
    p.set_defaults(func=cmd_explore)

    p = sub.add_parser("run", help="run one experiment")
    p.add_argument("--model", choices=sorted(MODEL_DEFAULTS), required=True)
    p.add_argument("--subset", choices=SUBSETS, default="FD001")
    p.add_argument("--seed", type=int, default=42, help="engine split + model init")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    p.add_argument("--healthy-cycles", type=int, default=50)
    p.add_argument("--rolling-window", type=int, default=10)
    p.add_argument("--threshold-q", type=float, default=0.99)
    p.add_argument("--score-smoothing", type=int, default=1)
    p.add_argument("--evaluate-test", action="store_true", help="open the official test set")
    _add_model_options(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="aggregate saved runs over seeds")
    p.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    p.add_argument("--subset", nargs="*", choices=SUBSETS)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument(
        "--metric",
        nargs="+",
        default=["early_warning_auc", "median_lead_time", "false_alarm_unit_rate"],
    )
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
