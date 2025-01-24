import yfinance as yf
import numpy as np
import pandas as pd

class ScalpingAnalysis:
    def __init__(self):
        self.ticker = "ES=F"  # E-mini S&P 500 Futures ticker
        self.period = "1d"  # Fetch 1 day of data
        self.interval = "1m"  # 1-minute interval for intraday data

    def get_yahoo_data(self):
        """Fetch data from Yahoo Finance"""
        try:
            # Fetch the last 1 day of data at 1-minute intervals
            data = yf.download(self.ticker, period=self.period, interval=self.interval)
            return data
        except Exception as e:
            print(f"Error fetching data from Yahoo Finance: {e}")
            return pd.DataFrame()

    def _custom_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """Custom MACD calculation"""
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line.iloc[-1],
            'signal_line': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
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
        
        return rsi.iloc[-1]

    def get_market_sentiment(self):
        """Placeholder for fetching market sentiment (using Yahoo or custom sources)"""
        # Here we can implement sentiment analysis or fetch news using another method
        return {
            "news": [
                {"headline": "Stock Market News", "summary": "Market opens up with bullish momentum."}
            ]
        }

    def advanced_scalping_analysis(self):
        """Comprehensive scalping preparation analysis"""
        # Fetch data from Yahoo Finance
        historical_data = self.get_yahoo_data()

        if historical_data.empty:
            return "Unable to fetch market data"
        
        # Market sentiment (Placeholder logic)
        sentiment = self.get_market_sentiment()
        
        # Price analysis
        close_prices = historical_data['Close']
        
        # Volatility calculation
        returns = close_prices.pct_change()
        volatility = {
            'historical_volatility': returns.std() * np.sqrt(252) * 100,
            'current_daily_change': returns.iloc[-1] * 100
        }
        
        # Technical indicators (MACD and RSI)
        macd = self._custom_macd(close_prices)
        rsi = self._custom_rsi(close_prices)
        
        # Compile comprehensive analysis
        analysis = f"""🎯 Advanced Scalping Preparation Guide 📊

💰 MARKET DATA SOURCES:
- Yahoo Finance: {'✅ Loaded' if not historical_data.empty else '❌ Failed'}

📊 MARKET STRUCTURE:
- Current Price: ${close_prices.iloc[-1]:,.2f}

📈 MARKET METRICS:
- Historical Volatility: {volatility['historical_volatility']:,.2f}%
- Daily Price Change: {volatility['current_daily_change']:,.2f}%

🚀 TECHNICAL INDICATORS:
- MACD Line: {macd['macd_line']:,.4f}
- MACD Signal: {macd['signal_line']:,.4f}
- MACD Histogram: {macd['histogram']:,.4f}
- RSI: {rsi:,.2f}

🌐 MARKET SENTIMENT:
{chr(10).join(f'• {news["headline"]}' for news in sentiment.get('news', [])[:2])}

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

def send_discord_message(message):
    webhook_url = "https://discord.com/api/webhooks/1326703378687983627/jYdBBNSOwNJt6fCP42rzfZspgVAHk5ge4SIbAVS1o0PiOXu4CJ8xbZsxLpTrJqqFYJln"
    max_length = 2000
    for i in range(0, len(message), max_length):
        chunk = message[i:i+max_length]
        requests.post(webhook_url, json={"content": chunk})

def main():
    scalping_analysis = ScalpingAnalysis()
    analysis = scalping_analysis.advanced_scalping_analysis()
    send_discord_message(analysis)

if __name__ == "__main__":
    main()
