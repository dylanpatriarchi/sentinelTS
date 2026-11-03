"""Anomaly detectors sharing a common fit/score interface."""

from sentinelts.models.base import AnomalyDetector
from sentinelts.models.isolation_forest import IsolationForestDetector

__all__ = ["AnomalyDetector", "IsolationForestDetector"]
