import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
import base64
import datetime

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1332276762603683862/aKE2i67QHm-1XR-HsMcQylaS0nKTS4yCVty4-jqvJscwkr6VRTacvLhP89F-4ABFDoQw"

class InradayMarketAnalysis:
    def __init__(self):
        """Initialize analysis with intraday-specific configurations"""
        self.analysis_config = {
            'period': '1d',  # 1 day of data
            'interval': '1m',  # 1-minute intervals for precise scalping
        }
        self.allowed_symbols = ['ES=F']
        
        # Scalping-specific parameters
        self.scalping_config = {
            'volatility_threshold': 0.5,  # Percentage volatility for entry
            'volume_threshold': 1.5,  # Multiple of average volume
        }

    def fetch_market_data(self, symbol):
        """
        Fetch high-resolution market data for intraday trading
        
        Args:
            symbol (str): Stock/futures symbol to analyze
        
        Returns:
            tuple: (market data DataFrame, error message or None)
        """
        if symbol not in self.allowed_symbols:
            return None, f"Symbol {symbol} not allowed. Only ES Futures are permitted."
        
        try:
            # Fetch high-resolution market data
            data = yf.download(
                symbol, 
                period=self.analysis_config['period'], 
                interval=self.analysis_config['interval']
            )
            
            # Validate and process data
            if data.empty:
                return None, f"No data available for {symbol}"
            
            # Add additional technical columns
            data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
            data['EMA_30'] = data['Close'].ewm(span=30, adjust=False).mean()
            data['RSI'] = self._calculate_rsi(data['Close'], period=14)
            
            return data, None
        
        except Exception as e:
            return None, f"Data fetch error for {symbol}: {str(e)}"

    def _calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices (pd.Series): Price series
            period (int): RSI calculation period
        
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        
        # Make two series: one for lower closes and one for higher closes
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        
        # Use exponential moving average
        ma_up = up.ewm(com=period-1, adjust=True, min_periods=period).mean()
        ma_down = down.ewm(com=period-1, adjust=True, min_periods=period).mean()
        
        rsi = ma_up / ma_down
        rsi = 100.0 - (100.0 / (1.0 + rsi))
        
        return rsi

    def identify_scalping_signals(self, data):
        """
        Generate scalping-specific trading signals
        
        Args:
            data (pd.DataFrame): Market price data
        
        Returns:
            dict: Scalping signals and insights
        """
        if len(data) < 30:
            return {"status": "INSUFFICIENT_DATA"}
        
        # Latest data point
        latest = data.iloc[-1]
        
        # Volatility check
        price_std = data['Close'].std()
        avg_price = data['Close'].mean()
        volatility_pct = (price_std / avg_price) * 100
        
        # EMA Crossover
        ema_crossover = "BULLISH" if data['EMA_10'].iloc[-1] > data['EMA_30'].iloc[-1] else "BEARISH"
        
        # RSI Signals
        rsi = data['RSI'].iloc[-1]
        
        # Volume analysis
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].mean()
        volume_ratio = current_volume / avg_volume
        
        # Scalping Entry Conditions
        scalping_entry = {
            "long_signal": (
                ema_crossover == "BULLISH" and 
                rsi < 30 and 
                volatility_pct > self.scalping_config['volatility_threshold'] and
                volume_ratio > self.scalping_config['volume_threshold']
            ),
            "short_signal": (
                ema_crossover == "BEARISH" and 
                rsi > 70 and 
                volatility_pct > self.scalping_config['volatility_threshold'] and
                volume_ratio > self.scalping_config['volume_threshold']
            )
        }
        
        return {
            "status": "ACTIVE",
            "volatility": volatility_pct,
            "ema_trend": ema_crossover,
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "entry_signals": scalping_entry
        }

    def generate_scalping_report(self, symbol='ES=F'):
        """
        Generate a comprehensive scalping-focused market report
        
        Args:
            symbol (str): Stock/futures symbol to analyze
        
        Returns:
            dict: Comprehensive scalping market analysis
        """
        # Fetch market data
        data, data_error = self.fetch_market_data(symbol)
        if data_error:
            return {'error': data_error}
        
        # Analyze scalping signals
        scalping_signals = self.identify_scalping_signals(data)
        
        # Prepare report
        report = {
            'symbol': symbol.replace('=F', ''),
            'current_price': data['Close'].iloc[-1],
            'signals': scalping_signals
        }
        
        return report

def generate_scalping_discord_message(report):
    """
    Generate a Discord message with scalping insights
    
    Args:
        report (dict): Scalping market report
    
    Returns:
        str: Formatted Discord message
    """
    if 'error' in report:
        return f"❌ Market Analysis Error: {report['error']}"
    
    signals = report['signals']
    
    message = f"""🎯 UWS Scalping Insights: {report['symbol']} 
📊 Current Price: ${report['current_price']:.2f}

💡 Market Signals:
• EMA Trend: {'🟢 Bullish' if signals['ema_trend'] == 'BULLISH' else '🔴 Bearish'}
• RSI: {signals['rsi']:.2f} {'🟢 Oversold' if signals['rsi'] < 30 else '🔴 Overbought' if signals['rsi'] > 70 else '⚖️ Neutral'}
• Volatility: {signals['volatility']:.2f}%
• Volume: {signals['volume_ratio']:.2f}x Avg

🚦 Trading Signals:
• Long Entry: {'🟢 TRIGGERED' if signals['entry_signals']['long_signal'] else '❌ Not Ready'}
• Short Entry: {'🔴 TRIGGERED' if signals['entry_signals']['short_signal'] else '❌ Not Ready'}
"""
    return message

def main():
    # Initialize market analysis
    market_analyzer = InradayMarketAnalysis()
    
    # Analyze ES Futures
    symbols = ['ES=F']
    
    # Generate reports
    reports = [market_analyzer.generate_scalping_report(symbol) for symbol in symbols]
    
    # Prepare Discord messages
    for report in reports:
        message = generate_scalping_discord_message(report)
        send_discord_message(DISCORD_WEBHOOK_URL, message)

# Keeping the original send_discord_message function from the previous script
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

if __name__ == "__main__":
    main()
