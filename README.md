# sentinelTS

Industrial time-series anomaly detection on the NASA C-MAPSS turbofan dataset,
built with Python and PyTorch.

> Status: work in progress. No model results are reported yet — numbers will
> appear here only once measured on the held-out test engines.

## Roadmap

- [x] Dataset loading, anomaly labelling and leakage-safe evaluation protocol
- [x] Preprocessing + Isolation Forest baseline (validation only, see [docs/components/02_isolation_forest.md](docs/components/02_isolation_forest.md))
- [x] LSTM Autoencoder (validation only, see [docs/components/03_lstm_autoencoder.md](docs/components/03_lstm_autoencoder.md))
- [x] Transformer-based detector + causal score smoothing (validation only, see [docs/components/04_transformer.md](docs/components/04_transformer.md))
- [x] Command-line interface (`sentinelts download | explore | run | compare`)
- [ ] Final evaluation on unseen test engines (FD001–FD004)

## Quickstart

```bash
uv sync                      # Python 3.11–3.13, installs the `sentinelts` command
uv run sentinelts download   # fetch + SHA-256 check C-MAPSS into data/raw
uv run sentinelts explore    # dataset statistics
uv run pytest                # tests
```

## CLI

```bash
# One experiment = one model, one subset, one engine split (seed). Validation only.
uv run sentinelts run --model iforest     --subset FD001 --seed 0
uv run sentinelts run --model lstm_ae     --subset FD001 --seed 0 --rolling-window 0 --window 50
uv run sentinelts run --model transformer --subset FD001 --seed 0 --rolling-window 0 --window 50 --score-smoothing 10

# Mean ± std over seeds of every configuration saved under artifacts/
uv run sentinelts compare --subset FD001
uv run sentinelts compare --metric early_warning_auc pr_auc --json

uv run sentinelts run --help   # all options, model hyper-parameters included
```

Each run is stored in `artifacts/<model>/<subset>/<label>-<config hash>/seed<k>/`
with `metrics.json` (full config, metrics, git commit, training curves) and
per-cycle scores. Runs differing only by seed share the config hash, which is
how `compare` groups them. `--evaluate-test` opens the official test set and
must only be used once, with a frozen configuration (see the protocol).

## Repository layout

```
src/sentinelts/
  data/         download, loading, RUL→anomaly labels, engine-level splits, dataset statistics
  evaluation/   point-wise & event-wise metrics, threshold selection
  features/     per-operating-condition normalisation, causal rolling features, causal windows
  models/       detectors with a common fit/score interface: Isolation Forest, LSTM Autoencoder,
                Transformer forecaster
                + reusable PyTorch training loop with engine-level early stopping
  experiment.py end-to-end run following the protocol (test set only on request)
  runs.py       run records (config hash, git revision) and cross-seed summaries
  cli.py        `sentinelts` command
tests/          unit tests (+ integration tests that run if data/raw exists)
docs/           evaluation_protocol.md — read this first; components/ — one write-up per component
```

## Evaluation in one paragraph

C-MAPSS gives Remaining Useful Life, not anomaly labels: cycles with RUL ≤ 30 are
anomalous, RUL > 100 healthy, in-between ignored. Engines (never rows) are split
into fit/validation; models learn only from the first 50 cycles of fit engines;
thresholds come from validation healthy data; the official test files are opened
once, at the end. Details and rationale: [docs/evaluation_protocol.md](docs/evaluation_protocol.md).
