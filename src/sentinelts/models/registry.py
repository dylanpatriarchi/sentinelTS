"""Single place that knows how to build each detector from plain parameters.

The CLI, the experiment runner and the run records all go through here, so a
model name + a dict of parameters fully describes a detector.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sentinelts.models.base import AnomalyDetector
from sentinelts.models.isolation_forest import IsolationForestDetector
from sentinelts.models.lstm_autoencoder import LSTMAutoencoderDetector
from sentinelts.models.training import TrainingConfig
from sentinelts.models.transformer import TransformerDetector

# Default hyper-parameters per model. Keys are also the CLI option names.
MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "iforest": {"n_estimators": 200},
    "lstm_ae": {"window": 30, "hidden_size": 64, "latent_size": 8, "score_on": "window"},
    "transformer": {"window": 30, "d_model": 32, "nhead": 4, "num_layers": 2, "dropout": 0.1},
}
NEURAL_MODELS = {"lstm_ae", "transformer"}
TRAINING_DEFAULTS: dict[str, Any] = {"epochs": 100, "lr": 1e-3, "patience": 10, "device": "cpu"}


def model_params(model: str, values: dict[str, Any]) -> dict[str, Any]:
    """Pick the parameters relevant to ``model`` from ``values`` (falling back to defaults)."""
    if model not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown model {model!r}; choose from {sorted(MODEL_DEFAULTS)}")
    keys = dict(MODEL_DEFAULTS[model])
    if model in NEURAL_MODELS:
        keys |= TRAINING_DEFAULTS
    return {k: values.get(k, default) for k, default in keys.items()}


def build_factory(model: str, params: dict[str, Any], seed: int) -> Callable[[], AnomalyDetector]:
    """Return a zero-argument constructor, as expected by ``run_experiment``."""
    p = model_params(model, params)
    if model == "iforest":
        return lambda: IsolationForestDetector(n_estimators=p["n_estimators"], seed=seed)

    training = TrainingConfig(
        epochs=p["epochs"], lr=p["lr"], patience=p["patience"], device=p["device"]
    )
    if model == "lstm_ae":
        return lambda: LSTMAutoencoderDetector(
            window=p["window"],
            hidden_size=p["hidden_size"],
            latent_size=p["latent_size"],
            score_on=p["score_on"],
            training=training,
            seed=seed,
        )
    return lambda: TransformerDetector(
        window=p["window"],
        d_model=p["d_model"],
        nhead=p["nhead"],
        num_layers=p["num_layers"],
        dropout=p["dropout"],
        training=training,
        seed=seed,
    )
