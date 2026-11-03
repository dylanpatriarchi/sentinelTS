# Component 4 — Transformer forecaster and score smoothing

## Model: forecasting, not reconstruction

A Transformer autoencoder without a bottleneck can learn the identity:
self-attention lets each output position read its own input position, so
anomalies get reconstructed too. We use a **one-step-ahead forecaster**
instead (`models.TransformerForecaster`):

```
cycles t-W+1 .. t-1 (B, W-1, F) ─► Linear(F→32) + learned positional embedding
                                ─► 2 × pre-LN TransformerEncoderLayer (4 heads, FFN 64, dropout 0.1)
                                ─► last position ─► Linear(32→F) ─► x̂_t
score(t) = mean_f (x_t,f − x̂_t,f)²
```

The model never sees `x_t` when predicting it (a unit test checks that changing
`x_t` leaves the prediction unchanged), so it cannot copy. Training reuses the
loop from component 3, now with a pluggable objective (`ErrorFn`:
`reconstruction_error` or `forecast_error`) used for both the loss and the
score. LSTM-AE results were checked to be bit-identical after this refactor.

## New evaluation option: causal score smoothing

`ExperimentConfig.score_smoothing = k`: anomaly scores are replaced by the mean
of the last k scores of the same engine **before** thresholding. It is
model-agnostic, so it can be applied identically to every detector.

## Validation results

Validation engines only; the official test set has **not** been used.
Same 3 engine splits (seeds 0, 1, 42) for every model.

```bash
uv run python scripts/run_experiment.py --model transformer --subset FD001 --rolling-window 0 --window 50 --score-smoothing 10 --seed 0
```

Mean ± std over the 3 splits (measured 2026-11-03), k = score smoothing:

| Subset | Model | k | Early-warning AUC | Median lead time | Engines with ≥1 false alarm |
|--------|-------|--:|------------------:|-----------------:|----------------------------:|
| FD001 | Isolation Forest, w=0      | 1  | 0.810 ± 0.032 | 65.0 ± 9.2  | 0.32 ± 0.08 |
| FD001 | Isolation Forest, w=0      | 10 | 0.821 ± 0.038 | 56.5 ± 6.2  | 0.08 ± 0.03 |
| FD001 | Isolation Forest, w=10     | 1  | 0.798 ± 0.042 | 55.8 ± 7.5  | 0.10 ± 0.05 |
| FD001 | LSTM-AE, W=50              | 1  | **0.886 ± 0.011** | 59.0 ± 2.3 | 0.13 ± 0.03 |
| FD001 | LSTM-AE, W=50              | 10 | 0.868 ± 0.012 | 53.5 ± 2.8  | 0.10 ± 0.05 |
| FD001 | Transformer, W=30          | 1  | 0.734 ± 0.029 | 69.8 ± 6.0  | 0.60 ± 0.05 |
| FD001 | Transformer, W=50          | 1  | 0.772 ± 0.025 | 76.3 ± 0.6  | 0.63 ± 0.03 |
| FD001 | Transformer, W=50          | 10 | 0.871 ± 0.034 | 68.5 ± 10.7 | 0.42 ± 0.03 |
| FD002 | Isolation Forest, w=0      | 1  | 0.773 ± 0.010 | 67.8 ± 2.9  | 0.31 ± 0.05 |
| FD002 | Isolation Forest, w=0      | 10 | 0.787 ± 0.014 | 55.8 ± 6.8  | 0.12 ± 0.02 |
| FD002 | Isolation Forest, w=10     | 1  | **0.800 ± 0.011** | 59.8 ± 2.6 | 0.19 ± 0.05 |
| FD002 | LSTM-AE, W=50              | 1  | 0.758 ± 0.007 | 43.0 ± 0.9  | 0.08 ± 0.05 |
| FD002 | LSTM-AE, W=50              | 10 | 0.741 ± 0.011 | 38.7 ± 1.6  | 0.06 ± 0.04 |
| FD002 | Transformer, W=30          | 1  | 0.679 ± 0.009 | 59.2 ± 2.3  | 0.48 ± 0.02 |
| FD002 | Transformer, W=50          | 1  | 0.704 ± 0.003 | 65.8 ± 0.8  | 0.46 ± 0.06 |
| FD002 | Transformer, W=50          | 10 | 0.774 ± 0.006 | 51.7 ± 4.2  | 0.21 ± 0.02 |

Detection rate is 1.00 for every configuration on validation.

## What we learned

- **Per-cycle scores are dominated by sensor noise.** The raw Transformer score
  (error on one cycle) behaves like the LSTM-AE `last` score from component 3:
  early alarms but ~60% of engines with a false alarm. Averaging over 10 cycles
  lifts its early-warning AUC by +0.10 (FD001) and +0.07 (FD002), on every split.
- **Smoothing helps scores that are not already aggregated.** LSTM-AE window
  scores already average over W cycles; extra smoothing slightly hurts them.
  For Isolation Forest it mostly cuts false alarms.
- **No single winner.** FD001: LSTM-AE (W=50) is best and most stable;
  the smoothed Transformer is close on AUC but has ~4× the false-alarm engines.
  FD002 (6 regimes): Isolation Forest with rolling features is still best.
- **Architecture vs objective are confounded.** LSTM-AE uses reconstruction,
  the Transformer uses forecasting; this comparison does not tell which of the
  two factors drives the difference.
- **Selection bias grows.** Many configurations have now been compared on the
  same validation engines. Final claims must come from a single frozen run on
  the test files.
