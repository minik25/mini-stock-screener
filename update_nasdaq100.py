import pandas as pd
import requests
from io import StringIO

print("Downloading NASDAQ-100 list from QQQ holdings...")

url = "https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?audienceType=Investor&etfCode=QQQ&download=1"

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/csv,application/csv,text/plain,*/*",
}

resp = requests.get(url, headers=headers, timeout=60)
resp.raise_for_status()

text = resp.text

# Some responses may contain a few lines of text before the CSV header; pandas can still usually handle it.
df = pd.read_csv(StringIO(text))

# Find the ticker column
ticker_col = None
for c in df.columns:
    if "ticker" in str(c).lower() or "symbol" in str(c).lower():
        ticker_col = c
        break

if ticker_col is None:
    raise RuntimeError(f"Could not find ticker column. Columns: {list(df.columns)}")

tickers = (
    df[ticker_col]
    .dropna()
    .astype(str)
    .str.strip()
    .str.replace(".", "-", regex=False)
)
tickers = sorted(set([t for t in tickers.tolist() if t]))

pd.Series(tickers).to_csv("data/nasdaq100.csv", index=False, header=False)

print("Saved data/nasdaq100.csv:", len(tickers), "tickers")
print("First 10:", tickers[:10])
