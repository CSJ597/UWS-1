import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
import base64

# REPLACE THIS WITH YOUR ACTUAL DISCORD WEBHOOK URL
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1332276762603683862/aKE2i67QHm-1XR-HsMcQylaS0nKTS4yCVty4-jqvJscwkr6VRTacvLhP89F-4ABFDoQw"

class MarketAnalysis:
    def __init__(self):
        """Initialize analysis with default configurations"""
        self.analysis_config = {
            'period': '5d',
            'interval': '5m',
            'technical_indicators': {
                'bollinger_window': 20
            }
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
            return None, f"Symbol {symbol} not allowed. Only ES Futures and S&P 500 are permitted."
        
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

    def generate_technical_chart(self, data, symbol):
        """
        Generate a comprehensive technical analysis chart
        
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
        window = self.analysis_config['technical_indicators']['bollinger_window']
        
        middle_band = close_prices.rolling(window=window).mean()
        std_dev = close_prices.rolling(window=window).std()
        upper_band = middle_band + (std_dev * 2)
        lower_band = middle_band - (std_dev * 2)
        
        plt.plot(data.index, close_prices, label='Close Price', color='blue')
        plt.plot(data.index, middle_band, label='Middle Band', color='gray', linestyle='--')
        plt.plot(data.index, upper_band, label='Upper Band', color='red', linestyle=':')
        plt.plot(data.index, lower_band, label='Lower Band', color='green', linestyle=':')
        
        plt.title(f'{symbol} Technical Analysis')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
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
        
        # Safely convert to scalar values
        def safe_scalar(series):
            try:
                return float(series.iloc[-1]) if not series.empty else np.nan
            except Exception:
                return np.nan
        
        # Compute metrics
        analysis = {
            'symbol': symbol,
            'current_price': safe_scalar(close_prices),
            'daily_change': safe_scalar(returns) * 100,
            'volatility': float(np.std(returns.dropna()) * np.sqrt(252) * 100),
            'technical_chart': self.generate_technical_chart(data, symbol)
        }
        
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
        payload['file'] = {
            'chart.png': base64.b64decode(chart_base64)
        }
    
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
    report = "🚀 Market Analysis Report 📊\n\n"
    chart = None
    
    for analysis in analyses:
        if 'error' in analysis:
            report += f"❌ Error: {analysis['error']}\n\n"
            continue
        
        report += f"""
🔍 Symbol: {analysis['symbol']}
💰 Current Price: ${analysis['current_price']:.2f}
📈 Daily Change: {analysis['daily_change']:.2f}%
🌪️ Volatility: {analysis['volatility']:.2f}%
""" + "-"*50 + "\n\n"
        
        # Use the first available chart
        if not chart:
            chart = analysis['technical_chart']
    
    return report, chart

def main():
    # Validate webhook URL
    if DISCORD_WEBHOOK_URL == "https://discord.com/api/webhooks/1332276762603683862/aKE2i67QHm-1XR-HsMcQylaS0nKTS4yCVty4-jqvJscwkr6VRTacvLhP89F-4ABFDoQw":
        print("ERROR: Please replace DISCORD_WEBHOOK_URL with your actual Discord webhook URL!")
        return

    # Initialize market analysis
    market_analyzer = MarketAnalysis()
    
    # Analyze ES Futures and S&P 500
    symbols = ['ES=F']
    
    # Perform analyses
    analyses = [market_analyzer.analyze_market(symbol) for symbol in symbols]
    
    # Generate report with chart
    report, chart = generate_market_report(analyses)
    
    # Send to Discord
    send_discord_message(DISCORD_WEBHOOK_URL, report, chart)

if __name__ == "__main__":
    main()
