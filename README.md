# sentinelTS

Industrial time-series anomaly detection on the NASA C-MAPSS turbofan dataset,
built with Python and PyTorch.

> Status: work in progress. No model results are reported yet — numbers will
> appear here only once measured on the held-out test engines.

## Roadmap

- [x] Dataset loading, anomaly labelling and leakage-safe evaluation protocol
- [ ] Preprocessing + Isolation Forest baseline
- [ ] LSTM Autoencoder
- [ ] Transformer-based detector
- [ ] Interactive dashboard (anomaly scores and alarms)
- [ ] Final evaluation on unseen test engines (FD001–FD004)

## Quickstart

```bash
uv sync                                   # Python 3.11–3.13
uv run python -m sentinelts.data.download # downloads + checksums C-MAPSS into data/raw
uv run python scripts/explore_dataset.py  # dataset statistics
uv run pytest                             # tests
```

## Repository layout

```
src/sentinelts/
  data/         loading (cmapss.py), RUL→anomaly labels (labels.py), engine-level splits (splits.py)
  evaluation/   point-wise & event-wise metrics, threshold selection
scripts/        runnable analyses
tests/          unit tests (+ integration tests that run if data/raw exists)
docs/           evaluation_protocol.md — read this first
```

## Evaluation in one paragraph

C-MAPSS gives Remaining Useful Life, not anomaly labels: cycles with RUL ≤ 30 are
anomalous, RUL > 100 healthy, in-between ignored. Engines (never rows) are split
into fit/validation; models learn only from the first 50 cycles of fit engines;
thresholds come from validation healthy data; the official test files are opened
once, at the end. Details and rationale: [docs/evaluation_protocol.md](docs/evaluation_protocol.md).
