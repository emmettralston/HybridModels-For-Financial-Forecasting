"""Compatibility layer to keep notebook imports stable."""

from Data.data import get_benchmark_prices  # re-export for notebooks expecting the new package layout

__all__ = ["get_benchmark_prices"]
