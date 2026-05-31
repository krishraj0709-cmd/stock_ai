# data/ingestion/yfinance_connector.py
import yfinance as yf
import pandas as pd


def _fix_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df


def fetch_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "pe_ratio":    info.get("trailingPE"),
        "pb_ratio":    info.get("priceToBook"),
        "eps":         info.get("trailingEps"),
        "revenue":     info.get("totalRevenue"),
        "debt_equity": info.get("debtToEquity"),
        "roe":         info.get("returnOnEquity"),
        "market_cap":  info.get("marketCap"),
    }


def fetch_historical(ticker, period="5y", interval="1d"):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True
    )
    df = _fix_columns(df)
    df.dropna(inplace=True)
    return df


def fetch_multiple(tickers, period="1y"):
    result = {}
    for ticker in tickers:
        try:
            result[ticker] = fetch_historical(ticker, period=period)
            print(f"  Fetched {ticker}: {len(result[ticker])} rows")
        except Exception as e:
            print(f"  Failed {ticker}: {e}")
    return result


if __name__ == "__main__":
    df = fetch_historical("RELIANCE.NS", period="1mo")
    print(df.tail())