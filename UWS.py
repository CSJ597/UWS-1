
import os
import time
import pytz
import base64
import logging
import tempfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from io import BytesIO
from bs4 import BeautifulSoup

# === CONFIGURATION === #
CONFIG = {
    'symbol': 'ES=F',
    'period': '1d',
    'interval': '1m',
    'lookback_minutes': 60,
    'volatility_annualization_factor': 252,
    'run_hour': 21,
    'run_minute': 59,
    'discord_webhook_url': "https://discord.com/api/webhooks/1332904597848723588/cmauevZsGfVQ5u4zo9AHepkBU3dxHXrRT1swWFc0EoJ2O9WgGJIam202DXhpYbEIZi7o",
    'ai_api_key': "32760184b7ce475e942fde2344d49a68",
    'finlight_api_key': "sk_ec789eebf83e294eb0c841f331d2591e7881e39ca94c7d5dd02645a15bfc6e52",
    'timezone': 'US/Eastern'
}

# === LOGGING SETUP === #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === UTILITY FUNCTIONS === #
def wait_until_next_run():
    tz = pytz.timezone(CONFIG['timezone'])
    now = datetime.now(tz)
    target = now.replace(hour=CONFIG['run_hour'], minute=CONFIG['run_minute'], second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    while target.weekday() > 4:
        target += timedelta(days=1)
    sleep_seconds = (target - now).total_seconds()
    logging.info(f"Next run at {target.strftime('%Y-%m-%d %I:%M %p %Z')}")
    time.sleep(sleep_seconds)

def retry(func, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            logging.warning(f"Retry {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
    raise RuntimeError("Maximum retries exceeded")

def fetch_yfinance_data(symbol, period, interval):
    import yfinance as yf
    return yf.Ticker(symbol).history(period=period, interval=interval)

def classify_trend(data):
    if len(data) < 2:
        return "INSUFFICIENT DATA"
    try:
        close = data['Close']
        ma = close.rolling(20).mean()
        slope = np.polyfit(range(len(ma.dropna())), ma.dropna(), 1)[0]
        atr = (data['High'] - data['Low']).rolling(14).mean().iloc[-1]
        if abs(slope) < 0.01 and atr < 1:
            return "CONSOLIDATING"
        elif slope > 0.02:
            return "BULLISH MOMENTUM"
        elif slope < -0.02:
            return "BEARISH MOMENTUM"
        else:
            return "RANGING"
    except Exception as e:
        logging.error(f"Trend analysis failed: {e}")
        return "ERROR"

def generate_chart(data, symbol):
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#1e222d')
    ax.set_facecolor('#1e222d')
    df = data.reset_index()
    ax.plot(df.index, df['Close'], color='white', linewidth=2)
    ax.fill_between(df.index, df['Close'], df['Close'].min(), color='white', alpha=0.1)
    ax.set_title(f"{symbol} Price Action", color='white')
    ax.tick_params(colors='white')
    buf = BytesIO()
    plt.savefig(buf, format='png', facecolor='#1e222d', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def send_discord_message(content="", chart_base64=None):
    webhook_url = CONFIG['discord_webhook_url']
    username = "Underground Wall Street 🏦"
    avatar_url = "https://i.ibb.co/3N2NV0C/UWS-B-2.png"
    payload = {"username": username, "avatar_url": avatar_url, "content": content}
    if chart_base64:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(base64.b64decode(chart_base64))
            tmp.flush()
            files = {"file": ("chart.png", open(tmp.name, "rb"), "image/png")}
            requests.post(webhook_url, data={"username": username, "avatar_url": avatar_url}, files=files)
            os.unlink(tmp.name)
            return
    requests.post(webhook_url, json=payload)

# === MAIN ANALYSIS === #
def run_analysis():
    data = retry(lambda: fetch_yfinance_data(CONFIG['symbol'], CONFIG['period'], CONFIG['interval']))
    if data.empty:
        send_discord_message("❌ No market data available for analysis.")
        return
    close = data['Close']
    current_price = close.iloc[-1]
    change = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
    volatility = np.std(close.pct_change().dropna()) * np.sqrt(CONFIG['volatility_annualization_factor']) * 100
    trend = classify_trend(data)
    chart = generate_chart(data, CONFIG['symbol'])
    summary = f"""
📈 **{CONFIG['symbol']} Market Summary**

• Price: ${current_price:.2f}
• Daily Change: {change:+.2f}%
• Volatility: {volatility:.2f}%
• Trend: {trend}
    """
    send_discord_message(summary.strip())
    send_discord_message(chart_base64=chart)

# === SCHEDULER === #
def main():
    while True:
        try:
            wait_until_next_run()
            run_analysis()
            time.sleep(30)
        except Exception as e:
            logging.error(f"Unhandled error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
