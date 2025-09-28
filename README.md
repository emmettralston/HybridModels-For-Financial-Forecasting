# Hybrid Models for Financial Forecasting

This repository explores hybrid time-series models that combine classical ARIMA
forecasts with machine-learning corrections for residual patterns. The goal is
to provide a lightweight sandbox for experimenting with hybrid pipelines on
financial price data such as equity indexes and cryptocurrencies.

## Repository Structure

- `hybrid_ts/` – Reusable Python package that implements the hybrid ARIMA + ML
  workflow, including utilities for splitting series and running backtests.
- `Data/` – Helper module that downloads and caches market data from Yahoo
  Finance so that notebooks have reproducible inputs.
- `Notebook/` – Jupyter notebook used for exploratory analysis and experimentation.
- `requirements.txt` – Minimal dependencies required to run the package and notebooks.
- `Research Paper Inspiration/` – Reference material that motivated this project.

## Quick Start

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run a short example in Python to train the hybrid model and compare it to a
   baseline ARIMA forecast:
   ```python
   from hybrid_ts.pipeline import backtest_hybrid
   from hybrid_ts.data import get_benchmark_prices

   sp500, btc = get_benchmark_prices()
   forecasts, metrics = backtest_hybrid(sp500, train_size=0.85)

   print(metrics)
   forecasts.tail()
   ```
   The `metrics` dictionary reports RMSE for the standalone ARIMA forecast versus
   the hybrid approach. The `forecasts` DataFrame contains the individual ARIMA,
   residual ML, and combined hybrid predictions alongside the actual values.

## Working with Notebooks

The `Notebook/exploratory_analysis.ipynb` file demonstrates the modelling steps
end-to-end. Launch Jupyter and open the notebook:

```bash
jupyter notebook Notebook/exploratory_analysis.ipynb
```

Yahoo Finance data is cached locally under `data/raw/` after the first download
so subsequent runs are faster. Delete cached CSV files if you need to refresh
prices.

## License

This project is released under the [MIT License](LICENSE).

