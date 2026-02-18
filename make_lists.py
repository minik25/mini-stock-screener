import os
import pandas as pd

print("Starting ticker list generation...")

os.makedirs("data", exist_ok=True)

def save(series, path):
    series = series.dropna().astype(str).str.strip()
    series = series[series != ""]
    series = series.str.replace(".", "-", regex=False)
    series = sorted(set(series.tolist()))
    pd.Series(series).to_csv(path, index=False, header=False)
    print(f"Saved {path}: {len(series)} tickers")


# S&P 500 from iShares IVV ETF
print("Downloading S&P 500 list...")
ivv = pd.read_csv(
    "https://www.ishares.com/us/products/239726/"
    "ishares-core-sp-500-etf/1467271812596.ajax"
    "?fileType=csv&fileName=IVV_holdings&dataType=fund",
    skiprows=9
)
save(ivv["Ticker"], "data/sp500.csv")


# NASDAQ-100 from Invesco QQQ ETF
print("Downloading NASDAQ-100 list...")
qqq = pd.read_csv(
    "https://www.invesco.com/us/financial-products/etfs/"
    "holdings/main/holdings/0?"
    "audienceType=Investor&etfCode=QQQ&download=1"
)

ticker_col = None
for col in qqq.columns:
    if "ticker" in col.lower() or "symbol" in col.lower():
        ticker_col = col
        break

if ticker_col is None:
    raise Exception("Could not find ticker column")

save(qqq[ticker_col], "data/nasdaq100.csv")

print("Done.")
