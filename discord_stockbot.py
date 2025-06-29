import os
import io
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import find_peaks
from sklearn.cluster import AgglomerativeClustering
import pandas_market_calendars as mcal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio

import discord
from discord.ext import commands

# ─── set up intents ───────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# ─── Chart Generator ────────────────────────────────────

def generate_sr_chart(ticker: str, asset_type: str) -> go.Figure:
    """
    Generate a 6-month support & resistance chart with moving averages and volume
    for a given ticker and asset type, removing weekends and exchange holidays.
    """
    # 1. Determine date range
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=200)

    # 2. Fetch OHLCV via Nasdaq /chart endpoint
    url = f"https://api.nasdaq.com/api/quote/{ticker}/chart"
    params = {
        "assetclass": asset_type,
        "fromdate":   start_date.strftime("%Y-%m-%d"),
        "todate":     end_date.strftime("%Y-%m-%d")
    }
    headers = {
        "Accept":     "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0"
    }
    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json().get("data", {}).get("chart")
    if data is None:
        raise ValueError(f"No chart data returned for {ticker} as {asset_type}")

    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["dateTime"])  # ensure datetime
    df = df.set_index("Date").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # 3. Trim to last 6 months
    end_dt = df.index.max()
    start_dt = end_dt - pd.DateOffset(months=6)
    df6 = df.loc[start_dt:end_dt].copy()
    df6.reset_index(inplace=True)  # bring Date back as a column

    # 4. Compute pivots & cluster levels
    highs, _ = find_peaks(df6["high"], distance=5, prominence=1)
    lows,  _ = find_peaks(-df6["low"],  distance=5, prominence=1)
    pivots = np.r_[
        df6["high"].iloc[highs].values,
        df6["low"].iloc[lows].values
    ]
    N_LEVELS = 6
    clust = AgglomerativeClustering(n_clusters=N_LEVELS, linkage="ward")
    labels = clust.fit_predict(pivots.reshape(-1, 1))
    levels = sorted(np.median(pivots[labels == i]) for i in range(N_LEVELS))

    # 5. Moving averages
    df6["EMA_8"]  = df6["close"].ewm(span=8, adjust=False).mean()
    df6["EMA_21"] = df6["close"].ewm(span=21, adjust=False).mean()
    df6["SMA_50"]  = df6["close"].rolling(50,  min_periods=1).mean()
    df6["SMA_100"] = df6["close"].rolling(100, min_periods=1).mean()
    df6["SMA_200"] = df6["close"].rolling(200, min_periods=1).mean()

    # 6. Compute holidays & rangebreaks
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(
        start_date=start_dt.date(),
        end_date=end_dt.date()
    )
    all_days = pd.date_range(start_dt, end_dt, freq='D').date
    holidays = sorted(set(all_days) - set(sched.index.date))
    hols_str = [d.strftime("%Y-%m-%d") for d in holidays]
    rangebreaks = [dict(bounds=["sat", "mon"]), dict(values=hols_str)]

    # 7. Build figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.02
    )
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df6["Date"], open=df6["open"], high=df6["high"],
            low=df6["low"], close=df6["close"], name="Price"
        ), row=1, col=1
    )
    # Volume
    fig.add_trace(
        go.Bar(
            x=df6["Date"], y=df6["volume"],
            marker_color='gray', showlegend=False
        ), row=2, col=1
    )
    # Moving Averages
    ma_cols = ["EMA_8", "EMA_21", "SMA_50", "SMA_100", "SMA_200"]
    colors  = ['blue', 'orange', 'green', 'purple', 'red']
    for col, color in zip(ma_cols, colors):
        fig.add_trace(
            go.Scatter(
                x=df6["Date"], y=df6[col],
                mode='lines', name=col,
                line=dict(width=1.5, color=color)
            ), row=1, col=1
        )
    # Support/Resistance lines
    shapes = []
    annotations = []
    for lvl in levels:
        shapes.append(dict(
            type="line", xref="paper", x0=0, x1=1,
            yref="y1", y0=lvl, y1=lvl,
            line=dict(dash="dash", width=1)
        ))
        annotations.append(dict(
            xref="paper", x=1.01, yref="y1", y=lvl,
            xanchor='left', text=f"{lvl:.2f}", showarrow=False,
            font=dict(size=11)
        ))

    # Apply shapes & annotations
    fig.update_layout(shapes=shapes, annotations=annotations)
    # Axis tweaks
    fig.update_yaxes(side="left", title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    # X-axis date formatting + rangebreaks
    for r in (1, 2):
        fig.update_xaxes(
            type="date", rangeslider_visible=False,
            rangebreaks=rangebreaks,
            row=r, col=1
        )
    # Final layout
    fig.update_layout(
        margin=dict(l=60, r=80, t=40, b=40),
        title=f"{ticker.upper()} – 6 Month Support & Resistance",
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h", x=1, xanchor="right",
            y=1.2, yanchor="top"
        )
    )
    return fig
    
@bot.command(name="chartsr")
async def _chartsr(ctx, ticker: str, asset_type: str = "stocks"):
    """Usage: !chartsr TICKER [stocks|etf|cryptocurrency]"""
    # normalize asset_type...
    msg = await ctx.send(f"Generating chart for `{ticker.upper()}` ({asset_type}) …")

    try:
        # 1) Generate the figure synchronously
        fig = generate_sr_chart(ticker, asset_type)

        # 2) Offload the blocking to_image into a threadpool
        loop = asyncio.get_running_loop()
        img_bytes = await loop.run_in_executor(
            None,
            lambda: fig.to_image(format="png", width=1000, height=600)
        )

        # 3) Send it back
        file = discord.File(io.BytesIO(img_bytes), filename=f"{ticker}.png")
        await ctx.send(file=file)

    except Exception as e:
        await ctx.send(f"⚠️ Error: {e}")

    finally:
        await msg.delete()
        
if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_TOKEN")
    bot.run(TOKEN)
