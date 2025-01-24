import yfinance as yf
import numpy as np
import pandas as pd
import requests

class ScalpingAnalysis:
    def get_yahoo_data(self, symbol='ES=F', period='1d', interval='5m'):
        """
        Fetch detailed market data from Yahoo Finance
        
        Args:
            symbol (str): Stock/futures symbol to analyze
            period (str): Time period for data retrieval
            interval (str): Candle interval
        
        Returns:
            pandas.DataFrame: Detailed market data
        """
        try:
            data = yf.download(symbol, period=period, interval=interval)
            return data
        except Exception as e:
            print(f"Yahoo Finance Data Fetch Error: {e}")
            return pd.DataFrame()
        
    def _custom_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """Custom MACD calculation"""
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd_line': float(macd_line.iloc[-1]) if len(macd_line) > 0 else np.nan,
            'signal_line': float(signal_line.iloc[-1]) if len(signal_line) > 0 else np.nan,
            'histogram': float(histogram.iloc[-1]) if len(histogram) > 0 else np.nan
        }

    def _custom_rsi(self, prices, periods=14):
        """Custom RSI calculation"""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=periods).mean()
        avg_loss = loss.rolling(window=periods).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return float(rsi.iloc[-1]) if not rsi.empty else np.nan

    def advanced_scalping_analysis(self, symbol='ES=F'):
        """Comprehensive scalping preparation analysis"""
        # Fetch data from Yahoo Finance
        data = self.get_yahoo_data(symbol)
        
        if data.empty:
            return f"Unable to fetch market data for {symbol}"
        
        # Price analysis
        close_prices = data['Close']
        
        # Volatility calculation
        returns = close_prices.pct_change()
        volatility = {
            'historical_volatility': float(returns.std() * np.sqrt(252) * 100),
            'current_daily_change': float(returns.iloc[-1] * 100)
        }
        
        # Technical indicators
        macd = self._custom_macd(close_prices)
        rsi = self._custom_rsi(close_prices)
        
        # Compile comprehensive analysis
        analysis = f"""🎯 Advanced Scalping Preparation Guide 📊

🔍 SYMBOL: {symbol}

📊 MARKET STRUCTURE:
- Current Price: ${close_prices.iloc[-1]:.2f}

📈 MARKET METRICS:
- Historical Volatility: {volatility['historical_volatility']:.2f}%
- Daily Price Change: {volatility['current_daily_change']:.2f}%

🚀 TECHNICAL INDICATORS:
- MACD Line: {macd['macd_line']:.4f}
- MACD Signal: {macd['signal_line']:.4f}
- MACD Histogram: {macd['histogram']:.4f}
- RSI: {rsi:.2f}

💡 SCALPING INSIGHTS:
{self._generate_scalping_insights(macd, rsi, volatility)}
"""
        return analysis

    def _generate_scalping_insights(self, macd, rsi, volatility):
        """Generate scalping insights based on indicators"""
        insights = []
        
        # MACD Trend Analysis
        if macd['histogram'] > 0:
            insights.append("🟢 Bullish Momentum: Consider long entries")
        elif macd['histogram'] < 0:
            insights.append("🔴 Bearish Momentum: Consider short entries")
        else:
            insights.append("⚪ Neutral Momentum: Wait for clear signal")
        
        # RSI Overbought/Oversold
        if rsi > 70:
            insights.append("⚠️ Potential Overbought: Risk of reversal")
        elif rsi < 30:
            insights.append("⚠️ Potential Oversold: Possible bounce")
        
        # Volatility Recommendation
        if volatility['historical_volatility'] > 2:
            insights.append("🔥 High Volatility: Tighter stop-losses recommended")
        else:
            insights.append("❄️ Low Volatility: Be cautious of false breakouts")
        
        return chr(10).join(insights)

def send_discord_message(webhook_url, message):
    """
    Send message to Discord webhook
    
    Args:
        webhook_url (str): Discord webhook URL
        message (str): Message to send
    """
    max_length = 2000
    for i in range(0, len(message), max_length):
        chunk = message[i:i+max_length]
        try:
            response = requests.post(webhook_url, json={"content": chunk})
            response.raise_for_status()  # Raise an exception for HTTP errors
        except Exception as e:
            print(f"Error sending Discord message: {e}")

def main():
    # Discord webhook URL
    webhook_url = "https://discord.com/api/webhooks/1332276762603683862/aKE2i67QHm-1XR-HsMcQylaS0nKTS4yCVty4-jqvJscwkr6VRTacvLhP89F-4ABFDoQw"
    
    # Initialize scalping analysis
    scalping_analysis = ScalpingAnalysis()
    
    # Example symbols to analyze
    symbols = ['ES=F', 'NQ=F', '^GSPC']
    
    # Collect all analyses
    full_analysis = ""
    
    for symbol in symbols:
        try:
            # Generate analysis for each symbol
            analysis = scalping_analysis.advanced_scalping_analysis(symbol)
            full_analysis += analysis + "\n\n" + "-"*50 + "\n\n"
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
    
    # Send analysis to Discord
    send_discord_message(webhook_url, full_analysis)
    print("Analysis sent to Discord successfully!")

if __name__ == "__main__":
    main()
