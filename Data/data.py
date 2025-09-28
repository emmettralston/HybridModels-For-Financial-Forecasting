"""Utilities for downloading and caching market data used across notebooks."""

from __future__ import annotations

import dataclasses as dc
import datetime as dt
import pathlib
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd
import yfinance as yf


_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
_DATA_DIR.mkdir(parents=True, exist_ok=True)


@dc.dataclass(frozen=True)
class PriceRequest:
    """Configuration describing a single ticker download."""

    ticker: str
    start: dt.date | dt.datetime | str | None = None
    end: dt.date | dt.datetime | str | None = None
    label: str | None = None
    auto_adjust: bool = True

    def with_overrides(self, **updates: Any) -> "PriceRequest":
        data = dc.asdict(self)
        data.update({k: v for k, v in updates.items() if k in data})
        return PriceRequest(**data)


_DEFAULT_BENCHMARKS: tuple[PriceRequest, ...] = (
    PriceRequest("^GSPC", start=dt.date(2002, 1, 1), end=dt.date(2023, 12, 31), label="sp500"),
    PriceRequest("BTC-USD", start=dt.date(2015, 1, 1), end=dt.date(2023, 12, 31), label="btc"),
)


def _ensure_datestr(value: dt.date | dt.datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dt.date, dt.datetime)):
        return value.strftime("%Y-%m-%d")
    msg = f"Unsupported date type: {type(value)!r}"
    raise TypeError(msg)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _cache_path(label: str, start: str | None, end: str | None) -> pathlib.Path:
    filename = f"{label}_{start or 'full'}_{end or 'latest'}.csv"
    return _DATA_DIR / filename


def _fetch_yahoo(
    ticker: str, start: str | None, end: str | None, *, auto_adjust: bool
) -> pd.Series:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        threads=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"Received empty dataset for ticker '{ticker}'")
    series = df[["Close"]].rename(columns={"Close": ticker}).squeeze("columns")
    return series


def _ensure_cached(request: PriceRequest, *, force_refresh: bool = False) -> pathlib.Path:
    label = request.label or _slug(request.ticker)
    start = _ensure_datestr(request.start)
    end = _ensure_datestr(request.end)
    path = _cache_path(label, start, end)
    if force_refresh or not path.exists():
        series = _fetch_yahoo(
            request.ticker,
            start=start,
            end=end,
            auto_adjust=request.auto_adjust,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        series.name = label
        series.to_csv(path)
    return path


def _load_series(path: pathlib.Path, label: str) -> pd.Series:
    series = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
    series = pd.to_numeric(series, errors="coerce").dropna()
    series.index.name = "Date"
    series.name = label
    return series


def _coerce_request(item: Any) -> PriceRequest:
    if isinstance(item, PriceRequest):
        return item
    if isinstance(item, str):
        return PriceRequest(item)
    if isinstance(item, Mapping):
        return PriceRequest(**item)
    msg = (
        "Price requests must be PriceRequest, ticker string, or mapping of"
        " parameters"
    )
    raise TypeError(msg)


def load_prices(
    requests: Sequence[PriceRequest | str | Mapping[str, Any]] | PriceRequest | str,
    *,
    force_refresh: bool = False,
    join: str = "outer",
) -> pd.DataFrame:
    """Download and align one or more tickers.

    Parameters
    ----------
    requests:
        Single request or sequence describing tickers to fetch.
    force_refresh:
        If ``True`` re-download even when a cached file exists.
    join:
        Join strategy passed to :func:`pandas.concat` when aligning series.
    """

    if isinstance(requests, (PriceRequest, str, Mapping)):
        request_items: Iterable[Any] = (requests,)
    else:
        request_items = requests

    series_list = []
    for item in request_items:
        req = _coerce_request(item)
        label = req.label or _slug(req.ticker)
        path = _ensure_cached(req, force_refresh=force_refresh)
        series_list.append(_load_series(path, label))

    frame = pd.concat(series_list, axis=1, join=join).sort_index()
    frame.index.name = "Date"
    return frame


def get_benchmark_prices(
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    force_refresh: bool = False,
    join: str = "outer",
) -> tuple[pd.Series, pd.Series]:
    """Return cached benchmark series for SP500 and BTC-USD.

    Parameters
    ----------
    overrides:
        Optional mapping keyed by ticker symbol that can override ``start``,
        ``end``, ``label``, or ``auto_adjust``.
    force_refresh:
        Force a re-download of the data even if the CSV cache exists.
    join:
        Join logic if the underlying series need alignment. Defaults to
        ``outer`` to retain the full history of each series.
    """

    overrides = overrides or {}
    adjusted = [
        request.with_overrides(**overrides.get(request.ticker, {}))
        for request in _DEFAULT_BENCHMARKS
    ]
    prices = load_prices(adjusted, force_refresh=force_refresh, join=join)
    series = [prices[col].dropna() for col in prices.columns]
    return tuple(series)


__all__ = [
    "PriceRequest",
    "get_benchmark_prices",
    "load_prices",
]
