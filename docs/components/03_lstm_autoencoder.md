# Component 3 — LSTM Autoencoder

## Model

Repeat-vector LSTM encoder–decoder (`models.LSTMAutoencoder`), in the spirit of
Malhotra et al. (2016):

```
window x (B, W, F) ─► LSTM encoder (H=64) ─► last hidden ─► Linear ─► z (8)   bottleneck
                    ─► Linear ─► repeat W times ─► LSTM decoder (H=64) ─► Linear ─► x̂ (B, W, F)
```

- Trained to reconstruct **healthy** windows only: windows ending at cycles 11–50
  of the fit engines, input = per-condition z-scored sensors (no rolling features).
- Anomaly score = reconstruction MSE, either over the whole window (`window`) or
  over the last cycle only (`last`).
- Adam, lr 1e-3, batch 128, gradient clipping 1.0, early stopping (patience 10)
  on 15% of the fit **engines** held out — never on validation engines, which
  are reserved for threshold and model selection.

## Pipeline change: windows for every detector

All detectors now receive `make_windows(...)`: one causal window per scored
cycle, left-padded with the engine's first observation. Isolation Forest uses
W = 1 and its results were checked to be **bit-identical** to component 2
(FD002, w=10, seed 42). Windows may reach into the 10-cycle warm-up, so every
model is still scored on exactly the same rows.

## Validation results

Validation engines only; the official test set has **not** been used.
Seeds 0, 1, 42 = three engine splits, the same for every model, so
comparisons can be made per seed (paired).

Reproduce one run:

```bash
uv run python scripts/run_experiment.py --model lstm_ae --subset FD001 --rolling-window 0 --window 50 --seed 0
```

Mean ± std over the 3 splits (measured 2026-11-03):

| Subset | Model | Early-warning AUC | Median lead time | Engines with ≥1 false alarm |
|--------|-------|------------------:|-----------------:|----------------------------:|
| FD001 | Isolation Forest, w=0       | 0.810 ± 0.032 | 65.0 ± 9.2 | 0.32 ± 0.08 |
| FD001 | Isolation Forest, w=10      | 0.798 ± 0.042 | 55.8 ± 7.5 | 0.10 ± 0.05 |
| FD001 | LSTM-AE, last, W=30         | 0.792 ± 0.041 | 75.0 ± 2.8 | 0.58 ± 0.06 |
| FD001 | LSTM-AE, window, W=10       | 0.783 ± 0.057 | 62.7 ± 5.3 | 0.25 ± 0.05 |
| FD001 | LSTM-AE, window, W=30       | 0.857 ± 0.035 | 61.8 ± 6.8 | 0.17 ± 0.03 |
| FD001 | LSTM-AE, window, W=50       | 0.886 ± 0.011 | 59.0 ± 2.3 | 0.13 ± 0.03 |
| FD002 | Isolation Forest, w=0       | 0.773 ± 0.010 | 67.8 ± 2.9 | 0.31 ± 0.05 |
| FD002 | Isolation Forest, w=10      | 0.800 ± 0.011 | 59.8 ± 2.6 | 0.19 ± 0.05 |
| FD002 | LSTM-AE, last, W=30         | 0.696 ± 0.010 | 58.5 ± 1.8 | 0.45 ± 0.07 |
| FD002 | LSTM-AE, window, W=10       | 0.715 ± 0.020 | 47.5 ± 3.1 | 0.15 ± 0.03 |
| FD002 | LSTM-AE, window, W=30       | 0.735 ± 0.011 | 43.5 ± 3.3 | 0.08 ± 0.02 |
| FD002 | LSTM-AE, window, W=50       | 0.758 ± 0.007 | 43.0 ± 0.9 | 0.08 ± 0.05 |

PR-AUC on RUL ≤ 30 vs RUL > 100 is ≥ 0.98 for every configuration (saturated,
see the protocol) and is omitted here; it is in each run's `metrics.json`.

Paired early-warning AUC per split:

| Subset | Seed | IF w=0 | IF w=10 | LSTM-AE W=30 | LSTM-AE W=50 |
|--------|-----:|-------:|--------:|-------------:|-------------:|
| FD001 | 0  | 0.810 | 0.802 | 0.869 | 0.898 |
| FD001 | 1  | 0.841 | 0.838 | 0.885 | 0.883 |
| FD001 | 42 | 0.778 | 0.754 | 0.818 | 0.877 |
| FD002 | 0  | 0.784 | 0.810 | 0.743 | 0.767 |
| FD002 | 1  | 0.765 | 0.789 | 0.738 | 0.753 |
| FD002 | 42 | 0.769 | 0.801 | 0.722 | 0.756 |

## What we learned

- **FD001: the temporal model helps.** LSTM-AE (window score, W ≥ 30) beats
  both Isolation Forest variants on every split, and longer windows help
  (W=10 < W=30 < W=50 on average).
- **FD002: it does not.** On every split the LSTM-AE has a lower early-warning
  AUC than Isolation Forest with rolling features, and alarms ~16 cycles later,
  although with fewer false alarms. This is a genuine negative result for now.
  Hypotheses to test (not yet tested):
  1. Per-condition z-scoring removes the regime mean but the *degradation
     signature* itself depends on the regime; in FD002 the regime changes
     every cycle, so a degraded window looks like a jagged pattern the AE
     partly reconstructs. Feeding the regime as an input may help.
  2. With 6 regimes, each regime has ~1/6 of the healthy data for its
     statistics; noisier normalisation inflates healthy reconstruction error.
- **`last` vs `window` scoring.** Scoring only the most recent cycle reacts
  earlier (FD001 lead time 75 vs 62) but triples false alarms: a single cycle is
  dominated by sensor noise. Averaging over the window is a built-in smoother —
  the same lead-time vs false-alarm trade-off seen with rolling features.
- **Healthy reconstruction loss plateaus around 0.45** (z-scored data): healthy
  cycles are largely white noise, which no autoencoder can compress. The model
  learns the slow structure; the remaining error is the noise floor.
- **Caveat — selection on validation.** We compared 4 LSTM-AE variants on the
  same validation engines. Picking the best of several configurations makes its
  validation score optimistic. Only the frozen test evaluation will tell how
  much.
