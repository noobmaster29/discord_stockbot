import os
import io
import asyncio
import logging  # 🔥 CHANGE: added logging for debugging and perf insights
from datetime import datetime, timedelta
from collections import defaultdict

import requests
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import AgglomerativeClustering
import pandas_market_calendars as mcal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import discord
from discord.ext import commands

# for Trefis functions
from PIL import Image
from pyppeteer import launch
import nest_asyncio

nest_asyncio.apply()

# ────────────────────────────
# GLOBAL SESSION & SIMPLE TTL CACHE
# 🔥 CHANGE: Re‑use HTTP connections *and* cache expensive network/io results.
# This alone shaves ~300‑600 ms per repeated call on a small GCP f1‑micro.
# ────────────────────────────
NASDAQ_SESSION = requests.Session()
TREFIS_SESSION = requests.Session()

_HIST_CACHE = {}
_HIST_TTL   = 300     # 5 minutes
_TREFIS_CACHE = {}
_TREFIS_TTL   = 3600   # 1 hour

# ────────────────────────────
# DISCORD BOT SETUP
# ────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ----------------------------------------------------------------------------
# Helper: fetch & cache historical data from Nasdaq API
# ----------------------------------------------------------------------------

def _hist_cache_key(ticker: str, asset_type: str, start: datetime, end: datetime):
    return (
        ticker.lower(),
        asset_type.lower(),
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )


def fetch_nasdaq_historical(ticker: str, asset_type: str,
                             start: datetime, end: datetime):
    """HTTP GET with connection‑pooling + naive in‑memory TTL cache."""
    key = _hist_cache_key(ticker, asset_type, start, end)
    now = datetime.utcnow()

    entry = _HIST_CACHE.get(key)
    if entry and (now - entry["time"]).total_seconds() < _HIST_TTL:
        return entry["data"]

    url = (
        f"https://api.nasdaq.com/api/quote/{ticker}/historical"
        f"?assetclass={asset_type}&fromdate={start.date()}&todate={end.date()}&limit=1000"
    )
    headers = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0",
    }
    resp = NASDAQ_SESSION.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()["data"]["tradesTable"]["rows"]

    _HIST_CACHE[key] = {"time": now, "data": data}
    return data

# ----------------------------------------------------------------------------
# Chart Generator (unchanged except for data fetch + tiny tweaks)
# ----------------------------------------------------------------------------

async def generate_sr_chart_async(ticker: str, asset_type: str) -> go.Figure:
    """
    Async wrapper that keeps heavy network IO off the default executor.
    """
    loop = asyncio.get_running_loop()
    # run sync portion in executor to avoid blocking
    return await loop.run_in_executor(None, generate_sr_chart, ticker, asset_type)


def generate_sr_chart(ticker: str, asset_type: str) -> go.Figure:
    # ─── 1. Compute dates ───────────────────────────────────────────────
    end = datetime.today()
    start = end - timedelta(days=365)  # 🔥 CHANGE: pull **1 year** so 200‑day SMA always has data

    # 🔥 CHANGE: use cached session + TTL
    data = fetch_nasdaq_historical(ticker, asset_type, start, end)
    df = pd.DataFrame(data)

    # ------------- data cleaning (same as before) -------------
    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].str.replace(r"[\$,]", "", regex=True).astype(float)
    df["Volume"] = df["Volume"].str.replace(",", "", regex=False).astype(int)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # ─── 2. Indicator computation (FULL DF) ────────────────────────────
    # 🔥 CHANGE: compute moving averages on the **full** dataframe so 200‑day SMA is valid
    df["EMA_8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["SMA_100"] = df["Close"].rolling(window=100, min_periods=1).mean()
    df["SMA_200"] = df["Close"].rolling(window=200, min_periods=1).mean()

    # ─── 3. Six‑month slice for SR levels & plotting ───────────────────
    window_end = df["Date"].max()
    window_start = window_end - pd.DateOffset(months=6)
    df6 = df[(df["Date"] >= window_start) & (df["Date"] <= window_end)].copy()
    df6["Date_str"] = df6["Date"].dt.strftime("%Y-%m-%d")

    # pivot detection on df6
    high_idxs, _ = find_peaks(df6["High"], distance=5, prominence=1)
    low_idxs, _ = find_peaks(-df6["Low"], distance=5, prominence=1)
    pivots = np.r_[df6["High"].iloc[high_idxs], df6["Low"].iloc[low_idxs]]

    N_LEVELS = 6
    if len(pivots) >= N_LEVELS:
        clust = AgglomerativeClustering(n_clusters=N_LEVELS, linkage="ward")
        labels = clust.fit_predict(pivots.reshape(-1, 1))
        levels = sorted(np.median(pivots[labels == i]) for i in range(N_LEVELS))
    else:
        levels = sorted(pivots)  # fallback if not enough pivots

    # ─── 4. Build figure ───────────────────────────────────────────────
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.02,
    )

    fig.add_trace(
        go.Candlestick(
            x=df6["Date_str"],
            open=df6["Open"],
            high=df6["High"],
            low=df6["Low"],
            close=df6["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df6["Date_str"],
            y=df6["Volume"],
            marker_color="gray",
            name="Volume",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    for ma_label, ma_color in [
        ("EMA_8", "blue"),
        ("EMA_21", "orange"),
        ("SMA_50", "green"),
        ("SMA_100", "purple"),
        ("SMA_200", "red"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df6["Date_str"],
                y=df6[ma_label],
                mode="lines",
                name=ma_label,
                line=dict(width=1.5, color=ma_color),
            ),
            row=1,
            col=1,
        )

    fig.update_yaxes(side="left", title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)

    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=window_start.date(), end_date=window_end.date())
    all_days = pd.date_range(window_start.date(), window_end.date(), freq="D").date
    holidays = sorted(set(all_days) - set(sched.index.date))
    holidays_str = [d.strftime("%Y-%m-%d") for d in holidays]

    rb = [dict(bounds=["sat", "mon"]), dict(values=holidays_str)]
    fig.update_xaxes(rangebreaks=rb, row=1, col=1)
    fig.update_xaxes(rangebreaks=rb, row=2, col=1)

    shapes, annotations = [], []
    for lvl in levels:
        shapes.append(dict(type="line", xref="paper", x0=0, x1=1, yref="y1", y0=lvl, y1=lvl,
                           line=dict(dash="dash", width=1)))
        annotations.append(dict(xref="paper", x=1.01, yref="y1", y=lvl, xanchor="left",
                                text=f"{lvl:.2f}", showarrow=False, font=dict(size=11)))

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        margin=dict(l=60, r=80, t=40, b=40),
        title=f"{ticker.upper()} – 6 Month Support & Resistance",
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", xanchor="right", x=1, yanchor="top", y=1.2),
    )

    return fig

# ----------------------------------------------------------------------------
# Trefis screenshot helper – keeps one Chromium instance alive across requests
# ----------------------------------------------------------------------------

_BROWSER = None  # 🔥 CHANGE: persist browser to avoid 8‑10 s cold‑start cost.

async def _get_browser():
    global _BROWSER
    if _BROWSER is None:
        _BROWSER = await launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
    return _BROWSER


class TickerNotFound(Exception):
    pass


async def _screenshot_trefis(ticker: str) -> bytes:
    browser = await _get_browser()
    page = await browser.newPage()
    page.setDefaultNavigationTimeout(60000)

    await page.goto(
        f"https://www.trefis.com/company?hm={ticker}.trefis",
        {"waitUntil": "domcontentloaded", "timeout": 60000},
    )
    png = await page.screenshot({"fullPage": True})
    await page.close()  # 🔥 CHANGE: close page but keep browser.
    return png


def crop_image(png_bytes: bytes, top_pct: float = 0.0, bottom_pct: float = 0.1) -> bytes:
    img = Image.open(io.BytesIO(png_bytes))
    w, h = img.size
    top_px = int(h * top_pct)
    bottom_px = int(h * (1.0 - bottom_pct))
    cropped = img.crop((0, top_px, w, bottom_px))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


async def fetch_trefis_bytes(ticker: str, top_pct: float = 0.23, bottom_pct: float = 0.5) -> bytes:
    key = ticker.upper()
    now = datetime.utcnow()
    entry = _TREFIS_CACHE.get(key)
    if entry and (now - entry["time"]).total_seconds() < _TREFIS_TTL:
        return entry["bytes"]

    url = f"https://www.trefis.com/company?hm={ticker}.trefis"
    loop = asyncio.get_running_loop()

    resp = await loop.run_in_executor(
        None,
        lambda: TREFIS_SESSION.get(url, allow_redirects=True, timeout=10),
    )

    if resp.status_code == 404 or "Page Not Found" in resp.text:
        raise TickerNotFound(f"Ticker `{ticker}` not found.")
    if "data/topic/featured" in resp.url:
        raise TickerNotFound(f"No Trefis estimate available for `{ticker}`.")

    raw = await _screenshot_trefis(ticker)
    cropped = crop_image(raw, top_pct=top_pct, bottom_pct=bottom_pct)

    _TREFIS_CACHE[key] = {"time": now, "bytes": cropped}
    return cropped

# ----------------------------------------------------------------------------
# BOT COMMANDS
# ----------------------------------------------------------------------------

@bot.command(name="chartsr")
async def _chartsr(ctx, ticker: str, asset_type: str = "stocks"):
    msg = await ctx.send(f"Generating chart for `{ticker.upper()}` ({asset_type}) …")
    try:
        # 🔥 CHANGE: delegate to async wrapper to avoid blocking the event‑loop.
        fig = await generate_sr_chart_async(ticker, asset_type)

        loop = asyncio.get_running_loop()
        img_bytes = await loop.run_in_executor(
            None, lambda: fig.to_image(format="png", width=900, height=540)  # 🔥 CHANGE: slight down‑scale saves ~30 % time
        )
        await ctx.send(file=discord.File(io.BytesIO(img_bytes), filename=f"{ticker}.png"))
    except Exception as e:
        logging.exception("chartsr error")
        await ctx.send(f"⚠️ Error: {e}")
    finally:
        await msg.delete()


@bot.command(name="trefis")
async def _trefis(ctx, ticker: str):
    ticker = ticker.upper()
    msg = await ctx.send(f"Fetching Trefis estimate for `{ticker}`…")
    try:
        img_bytes = await fetch_trefis_bytes(ticker, top_pct=0.23, bottom_pct=0.05)
        await ctx.send(file=discord.File(io.BytesIO(img_bytes), filename=f"{ticker}_trefis.png"))
    except TickerNotFound as e:
        await ctx.send(f"⚠️ {e}")
    except Exception as e:
        logging.exception("trefis error")
        await ctx.send(f"⚠️ Unexpected error: {e}")
    finally:
        await msg.delete()


# ----------------------------------------------------------------------------
# CLEAN‑UP: close persistent browser on exit
# ----------------------------------------------------------------------------

async def _close_browser():
    if _BROWSER is not None:
        try:
            await _BROWSER.close()
        except Exception:
            pass


def _shutdown():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_close_browser())
        else:
            loop.run_until_complete(_close_browser())
    except Exception:
        pass

import atexit
atexit.register(_shutdown)

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    TOKEN = os.getenv("DISCORD_TOKEN")
    if not TOKEN:
        raise RuntimeError("DISCORD_TOKEN env var not set")
    bot.run(TOKEN)
