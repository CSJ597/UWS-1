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
            'period': '1d',  # Change back to 1 day for valid data fetching.
            'interval': '1m',  # 1-minute granularity for scalping.
        }
        self.allowed_symbols = ['ES=F']

    def fetch_market_data(self, symbol):
        """
        Fetch comprehensive market data with error handling
        
        Args:
            symbol (str): Stock/futures symbol to analyze
        
        Returns:
            tuple: (market data DataFrame, error message or None)
        """
        if symbol not in self.allowed_symbols:
            return None, f"Symbol {symbol} not allowed. Only ES Futures are permitted."
        
        try:
            # Fetch detailed market data
            data = yf.download(
                symbol, 
                period=self.analysis_config['period'], 
                interval=self.analysis_config['interval']
            )
            
            # Validate data
            if data.empty:
                return None, f"No data available for {symbol}"
            
            return data, None
        
        except Exception as e:
            return None, f"Data fetch error for {symbol}: {str(e)}"

    def identify_market_trend(self, data):
        """
        Identify market trend (Trending or Ranging)
        
        Args:
            data (pd.DataFrame): Market price data
        
        Returns:
            str: Market trend classification
        """
        if len(data) < 2:
            return "INSUFFICIENT DATA"
        
        try:
            # Calculate price range and standard deviation
            price_range = float(data['High'].max() - data['Low'].min())
            
            avg_price = float(data['Close'].mean())
            
            # Prevent division by zero
            if avg_price == 0:
                return "UNDEFINED"
            
            # Calculate standard deviation
            price_std = float(data['Close'].std())
            
            # Coefficient of variation to assess trend
            cv = (price_std / avg_price) * 100
            
            # Trend classification logic
            first_close = float(data['Close'].iloc[0])
            last_close = float(data['Close'].iloc[-1])
            
            if cv < 0.3:
                return "RANGING"
            elif last_close > first_close and price_range > 0:
                return "BULLISH TREND"
            elif last_close < first_close and price_range > 0:
                return "BEARISH TREND"
            else:
                return "RANGING"
        
        except Exception as e:
            return f"TREND ANALYSIS ERROR: {str(e)}"

    def generate_technical_chart(self, data, symbol):
        """
        Generate a comprehensive technical analysis chart for last 12 hours
        
        Args:
            data (pd.DataFrame): Market price data
            symbol (str): Stock/futures symbol
        
        Returns:
            str: Base64 encoded chart image
        """
        plt.figure(figsize=(12, 8))
        
        # Set blue background for the chart
        plt.style.use('dark_background')  # Use a dark style
        plt.gca().set_facecolor('blue')  # Explicitly set the chart background to blue

        # Price line in white
        close_prices = data['Close']
        est_index = data.index.tz_convert('US/Eastern')

        plt.plot(est_index, close_prices, label='Close Price', color='white', linewidth=2)

        # Adjust grid and titles
        plt.title('Market Analysis - E-Mini S&P 500', pad=20, color='white')
        plt.xlabel('Time (EST)', color='white')
        plt.ylabel('Price', color='white')
        plt.legend(facecolor='blue', edgecolor='white')
        plt.grid(color='white', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45, color='white')
        plt.yticks(color='white')
        
        # Format x-axis to show EST times
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%I:%M %p', tz=est_index.tz))

        plt.tight_layout()
        
        # Save plot to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='blue')
        plt.close()
        
        # Encode image to base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def analyze_market(self, symbol='ES=F'):
        """
        Comprehensive market analysis
        
        Args:
            symbol (str): Stock/futures symbol to analyze
        
        Returns:
            dict: Comprehensive market analysis results
        """
        # Fetch market data
        data, data_error = self.fetch_market_data(symbol)
        if data_error:
            return {'error': data_error}
        
        # Analyze the last 3 hours (180 minutes)
        last_data = data.tail(180)
        close_prices = last_data['Close']
        returns = close_prices.pct_change()

        try:
            analysis = {
                'symbol': 'ES',  # Display as ES instead of ES=F
                'current_price': float(close_prices.iloc[-1]),
                'daily_change': float(returns.iloc[-1] * 100),
                'volatility': float(np.std(returns.dropna()) * np.sqrt(252) * 100),
                'market_trend': self.identify_market_trend(data),
                'technical_chart': self.generate_technical_chart(data, 'ES'),
                'session_high': float(last_data['High'].max()),
                'session_low': float(last_data['Low'].min())
            }
        except Exception as e:
            return {'error': f'Analysis error: {str(e)}'}
        
        return analysis

def send_discord_message(webhook_url, message, chart_base64=None):
    """
    Send message to Discord webhook with optional image
    
    Args:
        webhook_url (str): Discord webhook URL
        message (str): Text message to send
        chart_base64 (str, optional): Base64 encoded image
    """
    payload = {"content": message}
    
    if chart_base64:
        # Prepare the image for Discord
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
    # Initialize market analysis
    market_analyzer = MarketAnalysis()
    
    # Analyze ES Futures
    symbols = ['ES=F']
    
    # Perform analyses
    analyses = [market_analyzer.analyze_market(symbol) for symbol in symbols]
    
    # Generate report
    for analysis in analyses:
        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
        else:
            print(f"Market Report: {analysis}")
            send_discord_message(DISCORD_WEBHOOK_URL, "Market Analysis Report", analysis['technical_chart'])

if __name__ == "__main__":
    main()
