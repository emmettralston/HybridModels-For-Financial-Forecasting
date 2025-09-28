import yfinance as yf, pandas as pd, pathlib, datetime as dt

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

def _fetch_yahoo(ticker: str, start, end) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, threads=False, progress=False)
    df = df[["Close"]].rename(columns={"Close": ticker})
    return df

def get_benchmark_prices():
    sp_path = _DATA_DIR / "sp500.csv"
    btc_path = _DATA_DIR / "btc.csv"
    if not sp_path.exists():
        _fetch_yahoo("^GSPC", "2002-01-01", "2023-12-31").to_csv(sp_path)
    if not btc_path.exists():
        _fetch_yahoo("BTC-USD", "2015-01-01", "2023-12-31").to_csv(btc_path)
        
    return (
        pd.read_csv(sp_path, index_col=0, parse_dates=True).astype(float, errors='ignore').dropna(),
        pd.read_csv(btc_path, index_col=0, parse_dates=True).astype(float, errors='ignore').dropna(),
    )