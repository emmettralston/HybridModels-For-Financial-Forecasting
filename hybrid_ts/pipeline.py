"""Hybrid ARIMA + ML pipeline components for financial forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


def train_test_split_series(
    series: pd.Series, *, train_size: float = 0.8
) -> tuple[pd.Series, pd.Series]:
    """Split a univariate series into train/test portions by index order."""

    series = series.dropna().astype(float)
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1")

    split_idx = int(len(series) * train_size)
    if split_idx <= 0 or split_idx >= len(series):
        raise ValueError("train_size results in empty train/test split")
    return series.iloc[:split_idx], series.iloc[split_idx:]


def _build_lag_matrix(series: pd.Series, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1")

    lagged = pd.concat(
        {f"lag_{i}": series.shift(i) for i in range(1, n_lags + 1)}, axis=1
    ).dropna()
    y = series.loc[lagged.index]
    return lagged.to_numpy(), y.to_numpy()


@dataclass
class HybridARIMAModel:
    """Simple ARIMA + ML residual model."""

    arima_order: tuple[int, int, int] = (5, 1, 0)
    residual_lags: int = 5
    ml_model_factory: Callable[[], RegressorMixin] | None = None

    arima_: ARIMA | None = field(init=False, default=None)
    ml_model_: RegressorMixin | None = field(init=False, default=None)
    _residual_history: pd.Series | None = field(init=False, default=None)

    def _make_ml_model(self) -> RegressorMixin:
        if self.ml_model_factory is not None:
            return self.ml_model_factory()
        return RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42)

    def fit(self, series: pd.Series) -> "HybridARIMAModel":
        series = series.dropna().astype(float)
        if len(series) <= self.residual_lags + 5:
            raise ValueError("Insufficient observations to fit hybrid model")

        self.arima_ = ARIMA(
            series,
            order=self.arima_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()

        residuals = self.arima_.resid.dropna()
        if len(residuals) <= self.residual_lags:
            raise ValueError("Residual history too short for ML component")

        X, y = _build_lag_matrix(residuals, self.residual_lags)
        model = self._make_ml_model()
        model.fit(X, y)

        self.ml_model_ = model
        self._residual_history = residuals.iloc[-self.residual_lags :]
        return self

    def forecast(self, steps: int) -> pd.DataFrame:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if self.arima_ is None or self.ml_model_ is None or self._residual_history is None:
            raise RuntimeError("Model must be fitted before forecasting")

        arima_forecast = self.arima_.forecast(steps=steps)
        residual_history = self._residual_history.to_list()
        predicted_residuals: list[float] = []

        for _ in range(steps):
            features = np.array(
                [residual_history[-i] for i in range(1, self.residual_lags + 1)]
            ).reshape(1, -1)
            predicted = float(self.ml_model_.predict(features)[0])
            residual_history.append(predicted)
            predicted_residuals.append(predicted)

        residual_series = pd.Series(
            predicted_residuals, index=arima_forecast.index, name="ml_residual"
        )
        hybrid_series = arima_forecast + residual_series

        return pd.concat(
            {
                "arima": arima_forecast,
                "ml_residual": residual_series,
                "hybrid": hybrid_series.rename("hybrid"),
            },
            axis=1,
        )


def backtest_hybrid(
    series: pd.Series,
    *,
    train_size: float = 0.8,
    arima_order: tuple[int, int, int] = (5, 1, 0),
    residual_lags: int = 5,
    ml_model_factory: Callable[[], RegressorMixin] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Back-test the hybrid model versus pure ARIMA on a hold-out set."""

    train, test = train_test_split_series(series, train_size=train_size)
    kwargs = {
        "arima_order": arima_order,
        "residual_lags": residual_lags,
    }
    if ml_model_factory is not None:
        kwargs["ml_model_factory"] = ml_model_factory

    model = HybridARIMAModel(**kwargs).fit(train)
    forecasts = model.forecast(steps=len(test))
    forecasts["actual"] = test.values

    arima_rmse = float(np.sqrt(mean_squared_error(test, forecasts["arima"])))
    hybrid_rmse = float(np.sqrt(mean_squared_error(test, forecasts["hybrid"])))

    metrics = {
        "arima_rmse": arima_rmse,
        "hybrid_rmse": hybrid_rmse,
    }

    return forecasts, metrics


__all__ = [
    "HybridARIMAModel",
    "backtest_hybrid",
    "train_test_split_series",
]
