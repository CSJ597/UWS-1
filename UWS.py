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

class MarketAnalysis:
    def __init__(self):
        """Initialize analysis with default configurations"""
        self.analysis_config = {
            'period': '1d',  # Changed to 1 day
            'interval': '2m',  # Changed from 5m to 2m
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
            
            # Get last 12 hours of data
            last_12_hours = data.tail(12 * 6)  # 6 intervals per hour * 12 hours
            
            return last_12_hours, None
        
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
            
            # Use .item() to convert Series to scalar
            avg_price = float(data['Close'].mean().item())
            
            # Prevent division by zero
            if avg_price == 0:
                return "UNDEFINED"
            
            # Calculate standard deviation
            price_std = float(data['Close'].std().item())
            
            # Coefficient of variation to assess trend
            cv = (price_std / avg_price) * 100
            
            # Trend classification logic
            first_close = float(data['Close'].iloc[0])
            last_close = float(data['Close'].iloc[-1])
            
            if cv < 0.5:  # Very low variation
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
        plt.style.use('seaborn-v0_8')
        
        # Price and Bollinger Bands
        close_prices = data['Close']
        window = min(20, len(close_prices))  # Adjust window size if data is less
        
        middle_band = close_prices.rolling(window=window).mean()
        std_dev = close_prices.rolling(window=window).std()
        upper_band = middle_band + (std_dev * 2)
        lower_band = middle_band - (std_dev * 2)
        
        # Convert index to EST timezone (data is already tz-aware)
        est_index = data.index.tz_convert('US/Eastern')
        
        plt.plot(est_index, close_prices, label='Close Price', color='blue')
        plt.plot(est_index, middle_band, label='Middle Band', color='gray', linestyle='--')
        plt.plot(est_index, upper_band, label='Upper Band', color='red', linestyle=':')
        plt.plot(est_index, lower_band, label='Lower Band', color='green', linestyle=':')
        
        plt.title('Underground Wall Street\n E-Mini S&P 500 TA', pad=20)
        plt.xlabel('Time (EST)')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Format x-axis to show EST times
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%I:%M %p', tz=est_index.tz))
        
        plt.tight_layout()
        
        # Save plot to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
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
        
        # Get additional market info from Yahoo Finance
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get previous day's close for reference
            prev_day = ticker.history(period='2d', interval='1d')
            prev_close = float(prev_day['Close'].iloc[0]) if len(prev_day) > 1 else None
        except Exception as e:
            info = {}
            prev_close = None
            print(f"Warning: Could not fetch additional market data: {e}")
        
        # Analyze data
        close_prices = data['Close']
        returns = close_prices.pct_change()
        current_price = float(close_prices.iloc[-1].item())
        
        # Calculate session high and low
        session_high = float(data['High'].max())
        session_low = float(data['Low'].min())
        
        # Compute metrics
        try:
            analysis = {
                'symbol': 'ES',  # Display as ES instead of ES=F
                'current_price': current_price,
                'daily_change': float(returns.iloc[-1].item() * 100),
                'volatility': float(np.std(returns.dropna()) * np.sqrt(252) * 100),
                'market_trend': self.identify_market_trend(data),
                'technical_chart': self.generate_technical_chart(data, 'ES'),
                'session_high': session_high,
                'session_low': session_low,
                'prev_close': prev_close,
                'volume': int(data['Volume'].sum()) if 'Volume' in data else None,
                'avg_volume': info.get('averageVolume', None),
                'description': info.get('shortName', 'E-mini S&P 500 Futures')
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

def generate_market_report(analyses):
    """
    Generate a comprehensive market report
    
    Args:
        analyses (list): List of market analyses
    
    Returns:
        tuple: Formatted market report and chart (if available)
    """
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    report = f"""
📈 UWS Market Update 📉
📅 {current_date}
📊 E-Mini S&P 500 Mar 25
{'─' * 15}
"""
    chart = None
    
    for analysis in analyses:
        if 'error' in analysis:
            report += f"❌ Error: {analysis['error']}\n\n"
            continue
        
        # Trading Insights
        volatility_status = "LOW" if analysis['volatility'] < 15 else "HIGH" if analysis['volatility'] > 30 else "MODERATE"
        
        # Calculate price position relative to day's range
        price_position = (analysis['current_price'] - analysis['session_low']) / (analysis['session_high'] - analysis['session_low']) * 100 if analysis['session_high'] != analysis['session_low'] else 50
        
        # Format price position description
        range_position = "NEAR HIGH 🔝" if price_position > 75 else "NEAR LOW 📉" if price_position < 25 else "MID-RANGE ↔️"
        
        # Calculate change from previous close if available
        prev_close_info = ""
        if analysis['prev_close']:
            change_from_prev = ((analysis['current_price'] - analysis['prev_close']) / analysis['prev_close']) * 100
            arrow = "📈" if change_from_prev > 0 else "📉"
            prev_close_info = f"📊 From Previous Close: {arrow} {change_from_prev:+.2f}%\n"
            
        # Determine trend emoji
        trend_emoji = "🔄" if analysis['market_trend'] == "RANGING" else "📈" if "BULLISH" in analysis['market_trend'] else "📉"
        
        # Determine momentum emoji
        momentum_emoji = "🚀" if abs(analysis['daily_change']) > 1 else "🔄"
        
        report += f"""
💵 PRICE ACTION
• Current: **${analysis['current_price']:.2f}** ({range_position})
• Range: **${analysis['session_low']:.2f} - ${analysis['session_high']:.2f}**
• Today's Move: **{analysis['daily_change']:+.2f}%** 
{prev_close_info}
📊 TECHNICAL SNAPSHOT
• Market Trend: {trend_emoji} **{analysis['market_trend']}**
• Volatility: 🌪️ **{analysis['volatility']:.2f}%** ({volatility_status})
• Range Position: 📍 **{price_position:.1f}%**

🎯 TRADING SIGNALS
• Momentum: {momentum_emoji} {'Building' if abs(analysis['daily_change']) > 1 else 'Consolidating'}
• Volatility: {'⚠️ High' if volatility_status == 'HIGH' else '✅ Favorable' if volatility_status == 'MODERATE' else '⚡ Low'}
"""
        
        if analysis['volume'] and analysis['avg_volume']:
            volume_ratio = (analysis['volume'] / analysis['avg_volume']) * 100
            volume_emoji = "🔥" if volume_ratio > 100 else "💤"
            report += f"• Volume: {volume_emoji} {volume_ratio:.1f}% of average\n"
            
        report += f"\n{'═' * 40}\n"
        
        # Use the first available chart
        if not chart:
            chart = analysis['technical_chart']
    
    report += "\n🔍 Summary: Review key metrics for trading decisions."
    return report, chart

def main():
    # Initialize market analysis
    market_analyzer = MarketAnalysis()
    
    # Analyze ES Futures
    symbols = ['ES=F']
    
    # Perform analyses
    analyses = [market_analyzer.analyze_market(symbol) for symbol in symbols]
    
    # Generate report with chart
    report, chart = generate_market_report(analyses)
    
    # Send to Discord
    send_discord_message(DISCORD_WEBHOOK_URL, report, chart)

if __name__ == "__main__":
    main()
