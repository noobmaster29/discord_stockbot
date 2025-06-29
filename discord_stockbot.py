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
from discord.ext import commands

# ─── Chart Generator ────────────────────────────────────
def generate_sr_chart(ticker: str, asset_type: str) -> go.Figure:
    # 1) Fetch raw JSON from Nasdaq
    end = datetime.today().date()
    start = end - timedelta(days=200)        # 200-day lookback for SMA
    url = (
        f"https://api.nasdaq.com/api/quote/{ticker}/historical"
        f"?assetclass={asset_type}&fromdate={start}&todate={end}&limit=1000"
    )
    headers = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0"
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    rows = resp.json()["data"]["tradesTable"]["rows"]

    # 2) Build DataFrame & clean types
    df = pd.DataFrame(rows).rename(columns={
        "date": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume"
    })
    for col in ["Open","High","Low","Close"]:
        df[col] = df[col].str.replace(r"[\$,]", "", regex=True).astype(float)
    df["Volume"] = df["Volume"].str.replace(",", "").astype(int)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 3) Slice last 6 months
    end_dt = df["Date"].max()
    start_dt = end_dt - pd.DateOffset(months=6)
    df6 = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()

    # 4) Pivot detection & clustering
    highs, _ = find_peaks(df6["High"], distance=5, prominence=1)
    lows,  _ = find_peaks(-df6["Low"], distance=5, prominence=1)
    pivots = np.r_[df6["High"].iloc[highs], df6["Low"].iloc[lows]]
    clust = AgglomerativeClustering(n_clusters=6, linkage="ward")
    labels = clust.fit_predict(pivots.reshape(-1,1))
    levels = sorted(np.median(pivots[labels==i]) for i in range(6))

    # 5) Moving averages
    df6["EMA_8"]  = df6["Close"].ewm(span=8,  adjust=False).mean()
    df6["EMA_21"] = df6["Close"].ewm(span=21, adjust=False).mean()
    df6["SMA_50"]  = df6["Close"].rolling(50,  min_periods=1).mean()
    df6["SMA_100"] = df6["Close"].rolling(100, min_periods=1).mean()
    df6["SMA_200"] = df6["Close"].rolling(200, min_periods=1).mean()

    # 6) Holidays & rangebreaks
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=start_dt.date(), end_date=end_dt.date())
    all_days = pd.date_range(start_dt.date(), end_dt.date(), freq="D").date
    hols = sorted(set(all_days) - set(sched.index.date))
    hols = [d.strftime("%Y-%m-%d") for d in hols]
    rb = [dict(bounds=["sat","mon"]), dict(values=hols)]

    # 7) Build Plotly figure
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75,0.25], vertical_spacing=0.02
    )
    fig.add_trace(
        go.Candlestick(
            x=df6["Date"], open=df6["Open"], high=df6["High"],
            low=df6["Low"], close=df6["Close"], name="Price"
        ), row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=df6["Date"], y=df6["Volume"],
               marker_color="gray", showlegend=False),
        row=2, col=1
    )
    # MAs:
    for label,color in [
        ("EMA_8","blue"),("EMA_21","orange"),
        ("SMA_50","green"),("SMA_100","purple"),
        ("SMA_200","red")
    ]:
        fig.add_trace(
            go.Scatter(
                x=df6["Date"], y=df6[label],
                mode="lines", name=label,
                line=dict(width=1.5,color=color)
            ), row=1, col=1
        )
    # Axes & rangebreaks
    fig.update_yaxes(side="left", title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=rb, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=rb, row=2, col=1)

    # SR lines + labels
    shapes, ann = [], []
    for lvl in levels:
        shapes.append(dict(
            type="line", xref="paper", x0=0, x1=1.0,
            yref="y1", y0=lvl, y1=lvl,
            line=dict(dash="dash", width=1)
        ))
        ann.append(dict(
            xref="paper", x=1.01, yref="y1", y=lvl,
            xanchor="left", text=f"{lvl:.2f}",
            showarrow=False, font=dict(size=11)
        ))
    fig.update_layout(
        shapes=shapes, annotations=ann,
        margin=dict(l=60,r=80,t=40,b=40),
        title=f"{ticker.upper()} – 6 Month Support & Resistance",
        template="plotly_white", showlegend=True,
        legend=dict(
            orientation="h", x=1, xanchor="right",
            y=1.2, yanchor="top"
        )
    )

    return fig

# ─── Discord Bot ────────────────────────────────────────────
bot = commands.Bot(command_prefix="!")

@bot.command(name="chartsr")
async def _chartsr(ctx, ticker: str, asset_type: str = "stocks"):
    """
    Usage:
      !chartsr TICKER            → defaults to stocks
      !chartsr TICKER etf        → ETF
      !chartsr TICKER cryptocurrency
    """
    # normalize & validate
    asset_type = asset_type.lower()
    if asset_type not in ("stocks", "etf", "cryptocurrency"):
        await ctx.send(f"⚠️ Unknown asset type `{asset_type}`, defaulting to `stocks`.")
        asset_type = "stocks"

    msg = await ctx.send(f"Generating chart for `{ticker.upper()}` ({asset_type}) …")
    try:
        fig = generate_sr_chart(ticker, asset_type)
        img = fig.to_image(format="png", width=1000, height=600)
        await ctx.send(file=discord.File(io.BytesIO(img), filename=f"{ticker}.png"))
    except Exception as e:
        await ctx.send(f"⚠️ Error: {e}")
    finally:
        await msg.delete()

if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_TOKEN")
    bot.run(TOKEN)

#MTEzMTk3Njk3MzE5MTc2MTk4MQ.GY0Aob.DkAbweVxnYrQgI79jTgJUAV9aPEWYz3xJI6ej4