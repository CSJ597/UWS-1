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
            'interval': '5m',
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
            last_12_hours = data.tail(12 * 12)  # 12 intervals per hour * 12 hours
            
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
        
        # Convert index to EST timezone
        est_index = data.index.tz_localize('UTC').tz_convert('US/Eastern')
        
        plt.plot(est_index, close_prices, label='Close Price', color='blue')
        plt.plot(est_index, middle_band, label='Middle Band', color='gray', linestyle='--')
        plt.plot(est_index, upper_band, label='Upper Band', color='red', linestyle=':')
        plt.plot(est_index, lower_band, label='Lower Band', color='green', linestyle=':')
        
        plt.title(f'{symbol} Technical Analysis (Last 12 Hours) - EST')
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
        
        # Analyze data
        close_prices = data['Close']
        returns = close_prices.pct_change()
        
        # Compute metrics
        try:
            analysis = {
                'symbol': symbol,
                'current_price': float(close_prices.iloc[-1].item()),
                'daily_change': float(returns.iloc[-1].item() * 100),
                'volatility': float(np.std(returns.dropna()) * np.sqrt(252) * 100),
                'market_trend': self.identify_market_trend(data),
                'technical_chart': self.generate_technical_chart(data, symbol)
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
    report = f"🚀 SCALPING INSIGHTS: 📊\nDate: {current_date}\n\n"
    chart = None
    
    for analysis in analyses:
        if 'error' in analysis:
            report += f"❌ Error: {analysis['error']}\n\n"
            continue
        
        # Trading Insights
        volatility_status = "LOW" if analysis['volatility'] < 15 else "HIGH" if analysis['volatility'] > 30 else "MODERATE"
        
        report += f"""
🔍 Symbol: {analysis['symbol']}
💰 Current Price: ${analysis['current_price']:.2f}
📈 Daily Change: {analysis['daily_change']:.2f}%
🌪️ Volatility: {analysis['volatility']:.2f}% ({volatility_status})

🎯 SCALPING INSIGHTS:
- Market Trend: {analysis['market_trend']}
- Volatility Level: {volatility_status}
- Price Action: {'Momentum Building' if abs(analysis['daily_change']) > 1 else 'Consolidating'}
""" + "-"*50 + "\n\n"
        
        # Use the first available chart
        if not chart:
            chart = analysis['technical_chart']
    
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
