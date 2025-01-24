import yfinance as yf
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

class MarketAnalysis:
    def __init__(self):
        """Initialize analysis with default configurations"""
        self.analysis_config = {
            'period': '5d',
            'interval': '5m',
            'technical_indicators': {
                'bollinger_window': 20,
                'stochastic_periods': 14
            }
        }

    def fetch_market_data(self, symbol):
        """
        Fetch comprehensive market data with error handling
        
        Args:
            symbol (str): Stock/futures symbol to analyze
        
        Returns:
            tuple: (market data DataFrame, error message or None)
        """
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

    def fetch_company_info(self, symbol):
        """
        Retrieve comprehensive company/symbol information
        
        Args:
            symbol (str): Stock/futures symbol
        
        Returns:
            dict: Company information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Safely extract key information
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A')
            }
        except Exception as e:
            print(f"Error fetching company info for {symbol}: {e}")
            return {}

    def fetch_recent_news(self, symbol):
        """
        Retrieve recent news for the symbol
        
        Args:
            symbol (str): Stock/futures symbol
        
        Returns:
            list: Recent news articles
        """
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            # Format news with key details
            formatted_news = [
                {
                    'title': article.get('title', 'Untitled'),
                    'publisher': article.get('publisher', 'Unknown'),
                    'link': article.get('link', '')
                } for article in news[:3]  # Limit to 3 recent news
            ]
            
            return formatted_news
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return []

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
        plt.style.use('seaborn')
        
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
        
        # Save plot to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
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
        analysis = {
            'symbol': symbol,
            'current_price': close_prices.iloc[-1],
            'daily_change': returns.iloc[-1] * 100,
            'volatility': np.std(returns.dropna()) * np.sqrt(252) * 100,
            'technical_chart': self.generate_technical_chart(data, symbol),
            'company_info': self.fetch_company_info(symbol),
            'news': self.fetch_recent_news(symbol)
        }
        
        return analysis

def generate_discord_report(analyses):
    """
    Generate a comprehensive Discord report
    
    Args:
        analyses (list): List of market analyses
    
    Returns:
        str: Formatted Discord report
    """
    report = "🚀 Market Analysis Report 📊\n\n"
    
    for analysis in analyses:
        if 'error' in analysis:
            report += f"❌ Error: {analysis['error']}\n\n"
            continue
        
        report += f"""
🔍 Symbol: {analysis['symbol']}
💰 Current Price: ${analysis['current_price']:.2f}
📈 Daily Change: {analysis['daily_change']:.2f}%
🌪️ Volatility: {analysis['volatility']:.2f}%

ℹ️ Company Details:
• Name: {analysis['company_info'].get('name', 'N/A')}
• Sector: {analysis['company_info'].get('sector', 'N/A')}
• Market Cap: {analysis['company_info'].get('market_cap', 'N/A')}

🗞️ Recent News:
"""
        for news in analysis['news']:
            report += f"• {news['title']} (via {news['publisher']})\n"
        
        report += "\n" + "-"*50 + "\n\n"
    
    return report

def main():
    # Initialize market analysis
    market_analyzer = MarketAnalysis()
    
    # Symbols to analyze
    symbols = ['ES=F', '^GSPC', 'AAPL']
    
    # Perform analyses
    analyses = [market_analyzer.analyze_market(symbol) for symbol in symbols]
    
    # Generate report
    report = generate_discord_report(analyses)
    
    # Optional: Send to Discord or print
    print(report)
    
    # Uncomment and replace with your webhook if you want to send to Discord
    # send_discord_message(webhook_url, report)

if __name__ == "__main__":
    main()
