import requests
import numpy as np
import pandas as pd
import yfinance as yf

class ScalpingAnalysis:
    def __init__(self):
        # API keys (if using additional APIs, can be added here later)
        self.finnhub_api_key = "your_finnhub_api_key_here"
        
    def get_yahoo_data(self):
        """Fetch data from Yahoo Finance"""
        try:
            data = yf.download('ES=F', period='1d', interval='5m')
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
            'macd_line': macd_line.iloc[-1] if len(macd_line) > 0 else np.nan,
            'signal_line': signal_line.iloc[-1] if len(signal_line) > 0 else np.nan,
            'histogram': histogram.iloc[-1] if len(histogram) > 0 else np.nan
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
        
        return rsi.iloc[-1] if not rsi.empty else np.nan

    def get_market_sentiment(self):
        """Fetch market sentiment and news from Finnhub"""
        try:
            news_url = f"https://finnhub.io/api/v1/news?category=general&token={self.finnhub_api_key}"
            news_response = requests.get(news_url)
            news_data = news_response.json()
            
            # Safely extract top 3 news items
            news_data = news_data[:3] if isinstance(news_data, list) else []
            
            return {
                "news": [
                    {
                        "headline": news.get("headline", "No Headline"),
                        "summary": news.get("summary", "No Summary")
                    } for news in news_data
                ]
            }
        except Exception as e:
            print(f"Market Sentiment Error: {e}")
            return {"error": str(e)}

    def advanced_scalping_analysis(self):
        """Comprehensive scalping preparation analysis"""
        # Fetch data from Yahoo Finance
        data = self.get_yahoo_data()
        
        if data.empty:
            return "Unable to fetch market data"
        
        # Price analysis
        close_prices = data['Close']
        
        # Volatility calculation
        returns = close_prices.pct_change()
        volatility = {
            'historical_volatility': returns.std() * np.sqrt(252) * 100,
            'current_daily_change': returns.iloc[-1] * 100
        }
        
        # Technical indicators
        macd = self._custom_macd(close_prices)
        rsi = self._custom_rsi(close_prices)
        
        # Market sentiment
        sentiment = self.get_market_sentiment()
        
        # Compile comprehensive analysis
        analysis = f"""🎯 Advanced Scalping Preparation Guide 📊

💰 MARKET DATA SOURCES:
- Yahoo Finance: {'✅ Loaded' if not data.empty else '❌ Failed'}

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
    webhook_url = "your_discord_webhook_url_here"
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
