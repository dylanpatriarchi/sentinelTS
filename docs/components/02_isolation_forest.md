# Component 2 — Preprocessing and Isolation Forest baseline

## Pipeline

1. **Operating-condition normalisation** (`features.ConditionNormalizer`):
   KMeans on `op_1..op_3` (1 cluster for FD001/FD003, 6 for FD002/FD004), then
   per-condition z-score of each sensor using statistics of the healthy training
   prefix. Sensors with within-condition std < 0.01 are dropped (FD001 and
   FD002 keep the same 14: `s_1, s_5, s_6, s_10, s_16, s_18, s_19` are dropped).
2. **Causal rolling features** (`features.add_rolling_features`): rolling mean
   and std per sensor over the last *w* cycles of the same engine.
3. **Isolation Forest** (`models.IsolationForestDetector`), 200 trees, trained
   on cycles 11–50 of the fit engines; score = −`score_samples` (higher = more
   anomalous).
4. Threshold = 99th percentile of scores on healthy validation cycles.

## Validation results

All numbers below are on **validation engines** (20% of the official train
engines, split by engine). The official test set has **not** been used yet.
Three seeds = three different engine splits.

Reproduce:

```bash
for s in FD001 FD002; do for w in 0 10; do for seed in 42 0 1; do
  uv run python scripts/run_experiment.py --model iforest --subset $s --rolling-window $w --seed $seed
done; done; done
```

Mean ± std over seeds 0, 1, 42 (measured 2026-11-03):

| Subset | Rolling w | PR-AUC | Early-warning AUC | F1 | Detection rate | Median lead time (cycles) | Engines with ≥1 false alarm |
|--------|----------:|-------:|------------------:|---:|---------------:|--------------------------:|----------------------------:|
| FD001 | 0  | 0.998 ± 0.001 | 0.810 ± 0.032 | 0.978 ± 0.005 | 1.00 | 65.0 ± 9.2 | 0.32 ± 0.08 |
| FD001 | 10 | 0.999 ± 0.001 | 0.798 ± 0.042 | 0.979 ± 0.005 | 1.00 | 55.8 ± 7.5 | 0.10 ± 0.05 |
| FD002 | 0  | 0.993 ± 0.001 | 0.773 ± 0.010 | 0.962 ± 0.002 | 1.00 | 67.8 ± 2.9 | 0.31 ± 0.05 |
| FD002 | 10 | 0.999 ± 0.000 | 0.800 ± 0.011 | 0.981 ± 0.002 | 1.00 | 59.8 ± 2.6 | 0.19 ± 0.05 |

Random-detector PR-AUC (= anomaly ratio) ranges 0.23–0.28 across splits.

## What we learned

- **The near-failure task is saturated.** RUL ≤ 30 vs RUL > 100 is separated
  almost perfectly by a point-wise model. This metric cannot rank models; the
  early-warning AUC (≈0.77–0.84) and lead time / false-alarm trade-off can.
- **Smoothing trades lead time for fewer false alarms.** Rolling features cut
  the share of engines with a false alarm (FD001: 0.32 → 0.10) but alarms come
  ~9 cycles later.
- **Per-cycle vs per-engine false alarms.** A 1% per-cycle false-alarm rate
  still leaves 10–30% of engines with at least one false alarm, because each
  engine has ~90 healthy cycles. Operators experience the per-engine number.
- **Variance is large.** Only 20 (FD001) or 52 (FD002) validation engines:
  early-warning AUC moves by up to 0.06 between splits. Differences smaller
  than that between models are not meaningful without more seeds.
- **Bug caught by the metrics:** before the warm-up period, 18/19 false alarms
  of one run fell on cycle 1 (rolling std over one value = 0).
