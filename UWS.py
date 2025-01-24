import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
import base64
import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1332276762603683862/aKE2i67QHm-1XR-HsMcQylaS0nKTS4yCVty4-jqvJscwkr6VRTacvLhP89F-4ABFDoQw"

class MarketAnalysis:
    def __init__(self):
        """Initialize analysis with default configurations"""
        self.analysis_config = {
            'period': '1d',  # 1 day for valid data fetching
            'interval': '1m',  # 1-minute granularity for scalping
        }
        self.allowed_symbols = ['ES=F']

    def fetch_market_data(self, symbol):
        """Fetch comprehensive market data with error handling"""
        if symbol not in self.allowed_symbols:
            return None, f"Symbol {symbol} not allowed. Only ES Futures are permitted."

        try:
            data = yf.download(
                symbol,
                period=self.analysis_config['period'],
                interval=self.analysis_config['interval']
            )
            if data.empty:
                return None, f"No data available for {symbol}"
            return data, None
        except Exception as e:
            return None, f"Data fetch error for {symbol}: {str(e)}"

    def generate_technical_chart(self, data, symbol):
        """Generate a comprehensive technical analysis chart"""
        plt.figure(figsize=(12, 8))
        plt.style.use('seaborn-v0_8')

        # Set the background color
        plt.gca().set_facecolor("#a3c1ad")

        # Price and Bollinger Bands
        close_prices = data['Close']
        window = min(20, len(close_prices))  # Adjust window size if data is less
        middle_band = close_prices.rolling(window=window).mean()
        std_dev = close_prices.rolling(window=window).std()
        upper_band = middle_band + (std_dev * 2)
        lower_band = middle_band - (std_dev * 2)

        # Convert index to EST timezone
        est_index = data.index.tz_convert('US/Eastern')

        # Plot with white line
        plt.plot(est_index, close_prices, label='Close Price', color='white', linewidth=1.5)
        plt.plot(est_index, middle_band, label='Middle Band', color='gray', linestyle='--')
        plt.plot(est_index, upper_band, label='Upper Band', color='red', linestyle=':')
        plt.plot(est_index, lower_band, label='Lower Band', color='green', linestyle=':')

        # Updated title
        plt.title('Underground Wall Street\nE-Mini S&P 500 TA', pad=20, color='black')

        plt.xlabel('Time (EST)')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, color='black')
        plt.xticks(rotation=45)

        # Format x-axis to show EST times
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%I:%M %p', tz=est_index.tz))

        plt.tight_layout()

        # Save plot to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', facecolor="#a3c1ad")
        plt.close()

        # Encode image to base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def analyze_market(self, symbol='ES=F'):
        """Comprehensive market analysis"""
        # Fetch market data
        data, data_error = self.fetch_market_data(symbol)
        if data_error:
            return {'error': data_error}

        # Generate analysis results
        try:
            close_prices = data['Close']
            first_close = close_prices.iloc[0]
            last_close = close_prices.iloc[-1]
            price_range = data['High'].max() - data['Low'].min()
            cv = (np.std(close_prices) / np.mean(close_prices)) * 100
            cv = float(cv)  # Ensure cv is a scalar
            if cv < 0.3:
                market_trend = "RANGING"
            elif last_close > first_close and price_range > 0:
                market_trend = "BULLISH TREND"
            elif last_close < first_close and price_range > 0:
                market_trend = "BEARISH TREND"
            else:
                market_trend = "RANGING"

            analysis = {
                'symbol': 'ES',
                'current_price': data['Close'].iloc[-1].item(),
                'market_trend': market_trend,
                'technical_chart': self.generate_technical_chart(data, 'ES')
            }
        except Exception as e:
            return {'error': f'Analysis error: {str(e)}'}

        return analysis

def send_discord_message(webhook_url, message, chart_base64=None):
    """Send message to Discord webhook with optional image"""
    payload = {"content": message}

    if chart_base64:
        files = {
            'file': ('chart.png', base64.b64decode(chart_base64), 'image/png')
        }
        try:
            response = requests.post(webhook_url, data=payload, files=files)
            response.raise_for_status()
            print("Message with chart sent successfully to Discord!")
        except Exception as e:
            print(f"Error sending message to Discord: {e}")
    else:
        try:
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            print("Message sent successfully to Discord!")
        except Exception as e:
            print(f"Error sending message to Discord: {e}")

def main():
    market_analyzer = MarketAnalysis()
    symbols = ['ES=F']

    analyses = [market_analyzer.analyze_market(symbol) for symbol in symbols]
    for analysis in analyses:
        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
            continue

        message = f"Market Analysis for {analysis['symbol']}:\n" \
                  f"Current Price: ${analysis['current_price']:.2f}\n" \
                  f"Trend: {analysis['market_trend']}"
        chart = analysis.get('technical_chart')
        send_discord_message(DISCORD_WEBHOOK_URL, message, chart)

if __name__ == "__main__":
    main()
