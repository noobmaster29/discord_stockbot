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
    Fetches 6 months of daily OHLCV for a stock/ETF from Nasdaq's /chart endpoint,
    computes support/resistance levels and moving averages,
    and returns a Plotly Figure with weekends & holidays removed and proper date axes.
    """
    # 1) Date range
    today = datetime.today().date()
    start_200 = today - timedelta(days=200)

    # 2) Fetch via /chart endpoint
    url = f"https://api.nasdaq.com/api/quote/{ticker}/chart"
    params = {
        "assetclass": asset_type,
        "fromdate":   start_200.strftime("%Y-%m-%d"),
        "todate":     today.strftime("%Y-%m-%d")
    }
    headers = {"Accept": "application/json, text/plain, */*", "User-Agent": "Mozilla/5.0"}

    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    payload = resp.json().get("data", {})
    chart = payload.get("chart")
    if chart is None:
        # fallback: check historical tradesTable
        trades = payload.get("tradesTable", {}).get("rows")
        if trades is None:
            raise ValueError(f"No chart data for {ticker} ({asset_type})")
        df = pd.DataFrame(trades)
        df = df.rename(columns={ 'date':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume' })
        for c in ['Open','High','Low','Close']:
            df[c] = df[c].str.replace(r'[\$,]','',regex=True).astype(float)
        df['Volume'] = df['Volume'].str.replace(',','').astype(int)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        df = pd.DataFrame(chart)
        df['Date'] = pd.to_datetime(df['dateTime'])
        df = df.rename(columns={ 'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume' })
        for c in ['Open','High','Low','Close','Volume']:
            df[c] = df[c].astype(float)

    # 3) Filter last 6 months
    df = df.sort_values('Date').reset_index(drop=True)
    end6 = df['Date'].max()
    start6 = end6 - pd.DateOffset(months=6)
    df6 = df[(df['Date'] >= start6) & (df['Date'] <= end6)].copy()

    # 4) Pivots & clustering
    highs, _ = find_peaks(df6['High'], distance=5, prominence=1)
    lows,  _ = find_peaks(-df6['Low'], distance=5, prominence=1)
    pivots = np.r_[df6['High'].iloc[highs], df6['Low'].iloc[lows]]
    clust = AgglomerativeClustering(n_clusters=6, linkage='ward')
    labels = clust.fit_predict(pivots.reshape(-1,1))
    levels = sorted(np.median(pivots[labels==i]) for i in range(6))

    # 5) Moving averages
    df6['EMA_8']  = df6['Close'].ewm(span=8, adjust=False).mean()
    df6['EMA_21'] = df6['Close'].ewm(span=21,adjust=False).mean()
    df6['SMA_50']  = df6['Close'].rolling(50,min_periods=1).mean()
    df6['SMA_100'] = df6['Close'].rolling(100,min_periods=1).mean()
    df6['SMA_200'] = df6['Close'].rolling(200,min_periods=1).mean()

    # 6) Holidays rangebreaks
    nyse = mcal.get_calendar('NYSE')
    sched = nyse.schedule(start_date=start6.date(), end_date=end6.date())
    all_days = pd.date_range(start6.date(), end6.date(), freq='D').date
    hols = sorted(set(all_days) - set(sched.index.date))
    hols_str = [d.strftime('%Y-%m-%d') for d in hols]
    rb = [dict(bounds=['sat','mon']), dict(values=hols_str)]

    # 7) Build Plotly figure
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                        row_heights=[0.75,0.25],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df6['Date'],open=df6['Open'],high=df6['High'],
                                 low=df6['Low'],close=df6['Close'],name='Price'),row=1,col=1)
    fig.add_trace(go.Bar(x=df6['Date'],y=df6['Volume'],marker_color='gray',showlegend=False),row=2,col=1)
    for lbl,col in zip(['EMA_8','EMA_21','SMA_50','SMA_100','SMA_200'],
                       ['blue','orange','green','purple','red']):
        fig.add_trace(go.Scatter(x=df6['Date'],y=df6[lbl],mode='lines',
                                 name=lbl,line=dict(color=col,width=1.5)),row=1,col=1)
    # SR lines
    shapes, ann = [],[]
    for lvl in levels:
        shapes.append(dict(type='line',xref='paper',x0=0,x1=1,yref='y1',y0=lvl,y1=lvl,
                           line=dict(dash='dash',width=1)))
        ann.append(dict(xref='paper',x=1.01,yref='y1',y=lvl,xanchor='left',text=f"{lvl:.2f}",
                        showarrow=False,font=dict(size=11)))
    fig.update_layout(shapes=shapes,annotations=ann)
    # Axes
    fig.update_yaxes(side='left',title_text='Price',row=1,col=1)
    fig.update_yaxes(title_text='Volume',row=2,col=1)
    for r in (1,2):
        fig.update_xaxes(type='date',rangeslider_visible=False,rangebreaks=rb,row=r,col=1)
    # Final layout
    fig.update_layout(title=f"{ticker.upper()} – 6 Month Support & Resistance",template='plotly_white',
                      margin=dict(l=60,r=80,t=40,b=40),showlegend=True,
                      legend=dict(orientation='h',x=1,xanchor='right',y=1.2,yanchor='top'))
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
