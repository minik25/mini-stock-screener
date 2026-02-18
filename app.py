import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
import time
from yfinance.exceptions import YFRateLimitError
import random

st.set_page_config(page_title="Stock Screener", layout="wide")

# -----------------------------
# Index tickers
# -----------------------------
INDEX_MAP = {
    "NASDAQ Composite": "^IXIC",
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
}

# -----------------------------
# Load ticker lists (from local CSVs)
# -----------------------------
@st.cache_data
def load_sp500():
    return pd.read_csv("data/sp500.csv", header=None)[0].astype(str).tolist()

@st.cache_data
def load_nasdaq100():
    return pd.read_csv("data/nasdaq100.csv", header=None)[0].astype(str).tolist()

# -----------------------------
# Returns (used ONLY for indices tab)
# -----------------------------
def calc_returns(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return {"5D": np.nan, "MTD": np.nan, "YTD": np.nan, "1Y": np.nan, "All": np.nan}

    last = series.iloc[-1]
    out = {}

    # 5 trading days
    out["5D"] = (last / series.iloc[-6] - 1) * 100 if len(series) > 5 else np.nan

    # MTD
    month_start = pd.Timestamp(date.today().replace(day=1))
    mtd = series[series.index >= month_start]
    out["MTD"] = (last / mtd.iloc[0] - 1) * 100 if len(mtd) else np.nan

    # YTD
    year_start = pd.Timestamp(date.today().replace(month=1, day=1))
    ytd = series[series.index >= year_start]
    out["YTD"] = (last / ytd.iloc[0] - 1) * 100 if len(ytd) else np.nan

    # 1Y ~252 trading days
    out["1Y"] = (last / series.iloc[-252] - 1) * 100 if len(series) > 252 else np.nan

    # All-time
    out["All"] = (last / series.iloc[0] - 1) * 100

    return out
# -----------------------------
# Retry helpers (for rate limits)
# -----------------------------
def _sleep_jitter(base=0.6, jitter=0.6):
    time.sleep(base + random.random() * jitter)

def _retry_call(fn, tries=3):
    last_err = None
    for i in range(tries):
        try:
            return fn()
        except YFRateLimitError as e:
            last_err = e
            _sleep_jitter(1.0, 1.5)
        except Exception as e:
            last_err = e
            _sleep_jitter(0.5, 1.0)
    raise last_err

# -----------------------------
# Fetch price data (indices)
# -----------------------------
@st.cache_data(ttl=60*30)
def fetch_prices(tickers, period="2y"):
    data = _retry_call(lambda: yf.download(
        tickers,
        period=period,
        group_by="column",
        progress=False
    ))

    if data is None or data.empty:
        return pd.DataFrame()

    # Extract Close prices
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]]
        close.columns = tickers

    close.index = pd.to_datetime(close.index)
    return close

@st.cache_data(ttl=60*60)
def fetch_prices_safe(tickers, period="max", retries=3, pause=2.0):
    last_err = None

    for i in range(retries):
        try:
            df = yf.download(
                tickers=tickers,
                period=period,
                group_by="column",
                progress=False,
                threads=False,   # important
            )

            if df is None or df.empty:
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                close = df["Close"]
            else:
                close = df[["Close"]]
                close.columns = tickers

            close.index = pd.to_datetime(close.index)
            return close

        except YFRateLimitError:
            time.sleep(pause * (i + 1))
        except Exception:
            time.sleep(0.5)

    return pd.DataFrame()


# -----------------------------
# Fetch fundamentals (valuation + growth) for screeners
# -----------------------------

@st.cache_data(ttl=6*60*60)
def fetch_fundamentals(tickers):
    rows = []

    def pct(x):
        return x * 100 if x is not None and not pd.isna(x) else np.nan

    for t in tickers:
        try:
            info = yf.Ticker(t).info

            price = info.get("currentPrice", np.nan)
            high52 = info.get("fiftyTwoWeekHigh", np.nan)
            low52 = info.get("fiftyTwoWeekLow", np.nan)

            debt_to_equity = info.get("debtToEquity", np.nan)
            roe = info.get("returnOnEquity", np.nan)
            fcf = info.get("freeCashflow", np.nan)
            mktcap = info.get("marketCap", np.nan)

            pct_from_low = np.nan
            if pd.notna(price) and pd.notna(low52) and low52 != 0:
                pct_from_low = (price - low52) / low52 * 100

            pct_from_high = np.nan
            if pd.notna(price) and pd.notna(high52) and high52 != 0:
                pct_from_high = (high52 - price) / high52 * 100

            fcf_yield = np.nan
            if pd.notna(fcf) and pd.notna(mktcap) and mktcap != 0:
                fcf_yield = (fcf / mktcap) * 100

            rows.append({
                "Ticker": t,
                "Name": info.get("shortName", ""),
                "Sector": info.get("sector", ""),
                "Price": price,
                "MarketCap ($B)": (mktcap / 1e9) if pd.notna(mktcap) else np.nan,

                "PE (TTM)": info.get("trailingPE", np.nan),
                "Forward PE": info.get("forwardPE", np.nan),
                "PEG": info.get("pegRatio", np.nan),

                "Revenue YoY %": pct(info.get("revenueGrowth", None)),
                "EPS YoY %": pct(info.get("earningsGrowth", None)),

                "Gross Margin %": pct(info.get("grossMargins", None)),
                "Operating Margin %": pct(info.get("operatingMargins", None)),
                "Net Margin %": pct(info.get("profitMargins", None)),

                "FCF ($M)": (fcf / 1e6) if pd.notna(fcf) else np.nan,
                "FCF Yield %": fcf_yield,

                "ROE %": pct(roe),
                "Debt/Equity": debt_to_equity,

                "52W Low": low52,
                "52W High": high52,
                "% From 52W Low": pct_from_low,
                "% From 52W High": pct_from_high,
            })

        except Exception:
            rows.append({"Ticker": t})

    return pd.DataFrame(rows)


# -----------------------------
# UI
# -----------------------------
st.title("📈 Personalized Stock Screener")

tab1, tab2, tab3 = st.tabs(
    ["NASDAQ-100 Screener", "S&P 500 Screener", "Indices Dashboard"]
)

def screener_view(title, tickers, key_prefix):
    st.subheader(title)

    # -----------------------------
    # Load size selector (prevents 15+ min waits)
    # -----------------------------
    n = st.selectbox(
        "How many tickers to load (faster while testing)?",
        [25, 50, 100, 200, "All"],
        index=2,
        key=f"{key_prefix}_n"
    )

    if n != "All":
        tickers = tickers[:int(n)]

    run = st.button("Fetch / Refresh fundamentals", key=f"{key_prefix}_run")

    if not run:
        st.info("Choose number of tickers and filters, then click **Fetch / Refresh fundamentals**.")
        return

    # -----------------------------
    # Filters (institution-style)
    # -----------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        max_pe = st.number_input("Max PE (TTM)", value=40.0, step=5.0, key=f"{key_prefix}_pe")

    
    with c2:
        min_rev = st.number_input("Min Revenue YoY %", value=0.0, step=5.0, key=f"{key_prefix}_rev")

    with c3:
        min_eps = st.number_input("Min EPS YoY %", value=0.0, step=5.0, key=f"{key_prefix}_eps")

    with c4:
        max_from_high = st.slider("Max % From 52W High", 0, 80, 25, key=f"{key_prefix}_high")

        c6, c7, c8, c9 = st.columns(4)

    with c6:
        min_mktcap_b = st.number_input("Min Market Cap ($B)", value=10.0, step=5.0, key=f"{key_prefix}_mktcap")

    with c7:
        min_fcf_yield = st.number_input("Min FCF Yield %", value=0.0, step=1.0, key=f"{key_prefix}_fcfy")

    with c8:
        min_roe = st.number_input("Min ROE %", value=0.0, step=5.0, key=f"{key_prefix}_roe")

    with c9:
        max_de = st.number_input("Max Debt/Equity", value=999.0, step=0.5, key=f"{key_prefix}_de")

    # -----------------------------
    # Fetch fundamentals
    # -----------------------------
    with st.spinner(f"Fetching fundamentals for {len(tickers)} tickers..."):
        df = fetch_fundamentals(tickers)

    out = df.copy()

    if "PE (TTM)" in out.columns:
        out = out[(out["PE (TTM)"].isna()) | (out["PE (TTM)"] <= max_pe)]

    
    if "Revenue YoY %" in out.columns:
        out = out[(out["Revenue YoY %"].isna()) | (out["Revenue YoY %"] >= min_rev)]

    if "EPS YoY %" in out.columns:
        out = out[(out["EPS YoY %"].isna()) | (out["EPS YoY %"] >= min_eps)]

    if "% From 52W High" in out.columns:
        out = out[(out["% From 52W High"].isna()) | (out["% From 52W High"] <= max_from_high)]

    st.write(f"Matches: **{len(out)} / {len(df)}**")

    cols = [
        "Ticker","Name","Sector","Price","MarketCap ($B)",
        "PE (TTM)","Forward PE",
        "Revenue YoY %","EPS YoY %",
        "Gross Margin %","Operating Margin %","Net Margin %",
        "ROE %","Debt/Equity",
        "FCF ($M)","FCF Yield %",
        "52W Low","52W High",
        "% From 52W Low","% From 52W High"
    ]

    cols = [c for c in cols if c in out.columns]

    display_df = out[cols].copy()

    # identify numeric columns safely
    numeric_cols = display_df.select_dtypes(include=["number"]).columns.tolist()

    # round numeric values first
    display_df[numeric_cols] = display_df[numeric_cols].round(2)

    # apply styling
    styled_df = (
        display_df.style
        .set_properties(**{"text-align": "center"})
        .format({c: "{:.2f}" for c in numeric_cols}, na_rep="")
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True)



# -----------------------------
# NASDAQ-100 Screener
# -----------------------------
with tab1:
    tickers = load_nasdaq100()
    screener_view("NASDAQ-100 (Fundamentals Screener)", tickers, "ndx")

# -----------------------------
# S&P 500 Screener
# -----------------------------
with tab2:
    tickers = load_sp500()
    screener_view("S&P 500 (Fundamentals Screener)", tickers, "spx")

# -----------------------------
# Indices Dashboard (returns only)
# -----------------------------
with tab3:
    st.subheader("Major Indices (Levels + Returns)")

    idx_tickers = list(INDEX_MAP.values())
    prices = fetch_prices_safe(idx_tickers, period="max")

    rows = []
    for name, t in INDEX_MAP.items():
        if t not in prices.columns:
            continue
        series = prices[t].dropna()
        r = calc_returns(series)
        rows.append({
            "Index": name,
            "Level": series.iloc[-1] if len(series) else np.nan,
            "5D %": r["5D"],
            "MTD %": r["MTD"],
            "YTD %": r["YTD"],
            "1Y %": r["1Y"],
            "All-time %": r["All"],
        })

    # build dataframe
    df_idx = pd.DataFrame(rows)

    # numeric columns for 2-decimal formatting
    numeric_cols = df_idx.select_dtypes(include=["number"]).columns.tolist()

    # return columns to color
    return_cols = ["5D %", "MTD %", "YTD %", "1Y %", "All %", "All-time %"]
    return_cols = [c for c in return_cols if c in df_idx.columns]

    # function to color returns
    def color_returns(val):
        try:
            if pd.isna(val):
                return ""
            v = float(val)
            if v > 0:
                return "color: green;"
            if v < 0:
                return "color: red;"
            return ""
        except Exception:
            return ""

    # style dataframe
    styled_idx = (
        df_idx.style
        .set_properties(**{"text-align": "center"})
        .format({c: "{:.2f}" for c in numeric_cols}, na_rep="")
        .applymap(color_returns, subset=return_cols)
    )

    # display
    st.dataframe(styled_idx, use_container_width=True, hide_index=True)

    st.caption(
        "Returns are calculated from daily close prices: 5D (5 trading days), "
        "MTD, YTD, ~252 trading days for 1Y, and max history for all-time."

    )

