"""Shared evaluation helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class ModelResult:
    name: str
    r2: float
    rmse: float
    train_seconds: float

    def as_row(self) -> dict:
        return {
            "model": self.name,
            "r2": round(self.r2, 4),
            "rmse": round(self.rmse, 4),
            "train_seconds": round(self.train_seconds, 2),
        }


def fit_score(model, X_train, X_test, y_train, y_test, name: str) -> ModelResult:
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    pred = model.predict(X_test)
    return ModelResult(
        name=name,
        r2=r2_score(y_test, pred),
        rmse=float(np.sqrt(mean_squared_error(y_test, pred))),
        train_seconds=elapsed,
    )
