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
    # ─── 1. Compute dates ───────────────────────────────────────────────
    # 'end' = today’s date:
    end = datetime.today().date()  # returns a date object for today :contentReference[oaicite:0]{index=0}
    # 'start' = 200 trading days ago (to cover your 200-day SMA):
    start = end - timedelta(days=200)  # subtract a 200-day duration :contentReference[oaicite:1]{index=1}
    
    ticker = ticker
    asset_type = asset_type #stocks or etf, cryptocurrency wip
    
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
    data = resp.json()['data']['tradesTable']['rows']
    
    # … after you fetch `data = resp.json()['data']['tradesTable']['rows']` …
    
    df = pd.DataFrame(data)
    
    # 1) Standardize column names to match your other code
    df = df.rename(columns={
        'date':    'Date',
        'open':    'Open',
        'high':    'High',
        'low':     'Low',
        'close':   'Close',
        'volume':  'Volume'
    })
    
    # 2) Strip $ and commas from price columns, then convert to float
    for col in ['Open','High','Low','Close']:
        df[col] = (
            df[col]
            .str.replace(r'[\$,]', '', regex=True)  # remove $ and commas
            .astype(float)
        )
    
    # 3) Strip commas from Volume, convert to int
    df['Volume'] = (
        df['Volume']
        .str.replace(',', '', regex=False)
        .astype(int)
    )
    
    # 4) Convert Date to datetime, sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Now df has clean numeric OHLCV
    #df.head()
    
    # dynamic 6-month window
    end   = df['Date'].max()
    start = end - pd.DateOffset(months=6)
    df6   = df[(df['Date'] >= start) & (df['Date'] <= end)].copy()
    
    # ─── 2. Detect pivot highs & lows ─────────────────────────────────────────
    high_idxs, _ = find_peaks(df6['High'],  distance=5, prominence=1)
    low_idxs,  _ = find_peaks(-df6['Low'],  distance=5, prominence=1)
    pivots = np.r_[df6['High'].iloc[high_idxs], df6['Low'].iloc[low_idxs]]
    
    # ─── 3. Cluster into support/resistance levels ────────────────────────────
    N_LEVELS = 6
    clust = AgglomerativeClustering(n_clusters=N_LEVELS, linkage='ward')
    labels = clust.fit_predict(pivots.reshape(-1,1))
    levels = sorted(np.median(pivots[labels == i]) for i in range(N_LEVELS))
    
    # Calculate 8- and 21-day EMAs
    df6['EMA_8']  = df6['Close'].ewm(span=8,  adjust=False).mean()   # exponential smoothing :contentReference[oaicite:5]{index=5}
    df6['EMA_21'] = df6['Close'].ewm(span=21, adjust=False).mean()   # :contentReference[oaicite:6]{index=6}
    
    # Calculate 50-, 100-, and 200-day SMAs
    df6['SMA_50']  = df6['Close'].rolling(window=50,  min_periods=1).mean()   # simple moving average :contentReference[oaicite:7]{index=7}
    df6['SMA_100'] = df6['Close'].rolling(window=100, min_periods=1).mean()   # :contentReference[oaicite:8]{index=8}
    df6['SMA_200'] = df6['Close'].rolling(window=200, min_periods=1).mean()   # :contentReference[oaicite:9]{index=9}
    
    
    # ─── 4. Build figure with candles (row 1) + volume (row 2) ────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.02
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df6['Date'], open=df6['Open'],
            high=df6['High'], low=df6['Low'],
            close=df6['Close'], name='Price'
        ), row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=df6['Date'], y=df6['Volume'],
            marker_color='gray', name='Volume',
            showlegend=False
        ), row=2, col=1
    )
    
    # Overlay moving averages on the candlestick panel
    for ma_label, ma_color in [
        ('EMA_8',  'blue'),
        ('EMA_21', 'orange'),
        ('SMA_50', 'green'),
        ('SMA_100','purple'),
        ('SMA_200','red')
    ]:
        fig.add_trace(
            go.Scatter(
                x=df6['Date'],
                y=df6[ma_label],
                mode='lines',
                name=ma_label,
                line=dict(width=1.5, color=ma_color)
            ),
            row=1, col=1
        )  # Plotly Scatter on Candlestick :contentReference[oaicite:12]{index=12}
    
    
    # ─── 5. Axis tweaks ─────────────────────────────────────────────────────────
    # Price axis on left with title
    fig.update_yaxes(side="left", title_text="Price", row=1, col=1)
    # Volume axis on left
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    # Turn off the default range-slider
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
    
    # ─── 6. Compute holidays via pandas_market_calendars ───────────────────────
    nyse = mcal.get_calendar('NYSE')
    sched = nyse.schedule(start_date=start.date(), end_date=end.date())
    
    # all calendar days in window
    all_days = pd.date_range(start.date(), end.date(), freq='D').date
    # subtract trading days to get holidays/non-trading
    holidays = sorted(set(all_days) - set(sched.index.date))
    holidays_str = [d.strftime("%Y-%m-%d") for d in holidays]
    
    # apply rangebreaks for weekends + holidays
    rb = [
        dict(bounds=["sat", "mon"]),
        dict(values=holidays_str)
    ]
    fig.update_xaxes(rangebreaks=rb, row=1, col=1)
    fig.update_xaxes(rangebreaks=rb, row=2, col=1)
    
    # ─── 7. Draw SR lines + labels ─────────────────────────────────────────────
    shapes, annotations = [], []
    for lvl in levels:
        # dashed line across full width + 2% extra
        shapes.append(dict(
            type="line",
            xref="paper", x0=0, x1=1.0,
            yref="y1", y0=lvl, y1=lvl,
            line=dict(dash="dash", width=1)
        ))
        # numeric label just outside right edge
        annotations.append(dict(
            xref="paper", x=1.01, #1.025
            yref="y1", y=lvl,
            xanchor="left",
            text=f"{lvl:.2f}",
            showarrow=False,
            font=dict(size=11)
        ))
    
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        margin=dict(l=60, r=80, t=40, b=40),
        title= ticker.upper() + " – 6 Month Support & Resistance",
        template="plotly_white",
        showlegend=True
    )
    
    fig.update_layout(
        legend=dict(
            orientation="h",                # vertical list
            xanchor="right",                # anchor box’s right side at x
            x=1,                            # right edge of plot
            yanchor="top",               # anchor box’s bottom at y
            y=1.2                             # bottom of plotting area :contentReference[oaicite:10]{index=10}
        ),
        #margin=dict(b=80)                   # extra bottom margin to fit the legend
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
