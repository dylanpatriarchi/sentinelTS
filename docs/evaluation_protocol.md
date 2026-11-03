# Evaluation protocol

This document fixes *how* every model in sentinelTS is evaluated **before** any
model is built. Changing it later requires a documented reason, otherwise we
would be tuning the benchmark to the model.

## 1. Dataset: NASA C-MAPSS

Simulated turbofan engines (Saxena et al., PHM 2008). Each row = one flight
cycle of one engine: `unit, cycle, op_1..op_3` (operating settings) and
`s_1..s_21` (sensors). Four subsets:

| Subset | Conditions | Fault modes | Train engines | Test engines |
|--------|-----------:|------------:|--------------:|-------------:|
| FD001  | 1 | 1 (HPC) | 100 | 100 |
| FD002  | 6 | 1 (HPC) | 260 | 259 |
| FD003  | 1 | 2 (HPC, fan) | 100 | 100 |
| FD004  | 6 | 2 (HPC, fan) | 249 | 248 |

*(Counts measured by `sentinelts explore`; the NASA readme swaps the
FD004 train/test counts.)*

- **Train** series run until failure.
- **Test** series stop at a random point *before* failure; the RUL at the last
  observed cycle is provided in `RUL_FD00x.txt`.

Observations from `sentinelts explore` that drive the design:

- Minimum train lifetime is 128 cycles (FD001/2/4) and 145 (FD003).
- FD001: sensors `s_1, s_5, s_10, s_16, s_18, s_19` are near-constant
  (std < 1e-3); FD003 the same minus `s_10`. FD002/FD004 show no constant
  sensor *globally*, because the 6 operating conditions shift every sensor:
  they are constant only *within* a condition.
- Only a minority of test engines reach the anomalous zone (RUL <= 30):
  25/100 (FD001), 61/259 (FD002), 20/100 (FD003), 53/248 (FD004).

## 2. From RUL to anomaly labels

C-MAPSS has no anomaly labels. We derive them from RUL (`sentinelts.data.labels`):

| Label | Condition | Meaning |
|------:|-----------|---------|
| 0  | RUL > 100       | healthy |
| -1 | 30 < RUL <= 100 | ambiguous, **excluded from point-wise metrics** |
| 1  | RUL <= 30       | anomalous (degraded, near failure) |

Degradation onset differs per engine, so a hard healthy/anomalous boundary
would punish detectors for correctly spotting early degradation. Both
thresholds are parameters; sensitivity to them is an explicit experiment.

RUL is **ground truth only**. It is never a model input and never used to
select training rows.

## 3. Splits (leakage control)

```
train_FD00x.txt ──► split by ENGINE (seeded) ──► fit units (80%) ─► first 50 cycles ─► model training
                                             └─► val units  (20%) ─► full series   ─► threshold + model selection
test_FD00x.txt  ─────────────────────────────────────────────────────────────────────► FINAL hold-out (opened once)
```

1. **Split by engine, never by row.** Consecutive cycles are almost identical;
   a row-level split leaks near-duplicates into validation.
2. **Train on early life only.** The detector learns "normal" from the first
   50 cycles of each fit engine (`healthy_prefix`). Selection is by *cycle
   index*, which is known in production, not by RUL. Measured contamination of
   this prefix: 0.2–1.5% of rows (depending on subset) fall in the ambiguous zone, minimum RUL 78
   (95 for FD003) — none in the anomalous zone.
3. **All fitted preprocessing** (scalers, per-condition normalisation, sensor
   selection, PCA…) is fitted on the fit units' training rows only and then
   applied unchanged to val and test.
4. **Thresholds are chosen on validation healthy cycles** (`quantile_threshold`,
   default q = 0.99 ≈ 1% target false-alarm rate). Never on test.
5. **Sliding windows** (for LSTM/Transformer) are built *within* each engine
   and are causal: the score at cycle *t* uses only cycles <= *t*.
6. **The official test set is the final hold-out** and is used once per
   finalised model. Every result reported in the README states whether it is
   validation or test.

## 4. Metrics (`sentinelts.evaluation`)

**Point-wise** (one sample per labelled cycle, ambiguous dropped):
- PR-AUC (average precision) — headline, robust to class imbalance;
  always reported next to `anomaly_ratio`, the PR-AUC of a random detector.
- ROC-AUC — reported for comparability with literature.
- Precision / recall / F1 at the validation-chosen threshold.

**Event-wise** (one sample per engine) — what an operator cares about:
- `detection_rate`: fraction of engines reaching RUL <= 30 that raise >= 1
  alarm in the anomalous zone.
- `median_lead_time`: RUL at the first non-healthy alarm (higher = earlier).
- `false_alarm_unit_rate`: fraction of engines with >= 1 alarm while healthy.
- `false_alarm_point_rate`: alarms / healthy cycles.

**Early-warning AUC** (added in component 2, see §6): ROC-AUC separating
ambiguous cycles (30 < RUL <= 100) from healthy ones.

**Not used: point-adjusted F1.** It marks a whole anomalous segment as detected
if any single point is flagged; Kim et al. (AAAI 2022) show it lets random
scores rival published models.

## 5. Warm-up

Cycles 1..10 of every engine are neither used for training nor scored
(`ExperimentConfig.warmup_cycles`). Rolling features are undefined on an
incomplete window, and the value is fixed across all configurations so every
experiment is evaluated on identical rows. All engines live >= 128 cycles, so
only healthy cycles are removed.

## 6. Amendments log

| Date | Change | Reason (measured on validation, never on test) |
|------|--------|-----------------------------------------------|
| 2026-11-03 | Added warm-up (§5) | With rolling window 10, 18/19 false alarms of FD001 (seed 42) fell on cycle 1, where the rolling std of a single value is 0. |
| 2026-11-03 | Added early-warning AUC (§4) | Isolation Forest reaches PR-AUC >= 0.99 on RUL <= 30 vs RUL > 100: that metric is saturated and cannot rank models. Early detection is where they can differ. |

## 7. Definition of done

A model counts as evaluated only when it has (a) validation metrics used for
all decisions and (b) a single run on the untouched test files of FD001 (and
later FD002–FD004), with threshold and hyperparameters frozen beforehand.
