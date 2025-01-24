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
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands using Yahoo Finance style"""
        middle_band = prices.rolling(window=window).mean()
        std_dev = prices.rolling(window=window).std()
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        # Return latest values
        return {
            'middle_band': float(middle_band.iloc[-1]) if not middle_band.empty else np.nan,
            'upper_band': float(upper_band.iloc[-1]) if not upper_band.empty else np.nan,
            'lower_band': float(lower_band.iloc[-1]) if not lower_band.empty else np.nan
        }

    def _calculate_stochastic_oscillator(self, data, periods=14):
        """Calculate Stochastic Oscillator"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        lowest_low = low.rolling(window=periods).min()
        highest_high = high.rolling(window=periods).max()
        
        # %K line (Stochastic)
        k_line = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # %D line (Signal line)
        d_line = k_line.rolling(window=3).mean()
        
        return {
            'k_line': float(k_line.iloc[-1]) if not k_line.empty else np.nan,
            'd_line': float(d_line.iloc[-1]) if not d_line.empty else np.nan
        }

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
        
        # Safely extract scalar values
        current_price = float(close_prices.iloc[-1])
        historical_volatility = float(np.std(returns.dropna()) * np.sqrt(252) * 100)
        current_daily_change = float(returns.iloc[-1] * 100)
        
        # Technical indicators
        bollinger = self._calculate_bollinger_bands(close_prices)
        stochastic = self._calculate_stochastic_oscillator(data)
        
        # Compile comprehensive analysis
        analysis = f"""🎯 Advanced Scalping Preparation Guide 📊

🔍 SYMBOL: {symbol}

📊 MARKET STRUCTURE:
- Current Price: ${current_price:.2f}

📈 MARKET METRICS:
- Historical Volatility: {historical_volatility:.2f}%
- Daily Price Change: {current_daily_change:.2f}%

🚀 TECHNICAL INDICATORS:
- Bollinger Middle Band: ${bollinger['middle_band']:.2f}
- Bollinger Upper Band: ${bollinger['upper_band']:.2f}
- Bollinger Lower Band: ${bollinger['lower_band']:.2f}
- Stochastic %K: {stochastic['k_line']:.2f}
- Stochastic %D: {stochastic['d_line']:.2f}

💡 SCALPING INSIGHTS:
{self._generate_scalping_insights(bollinger, stochastic, {
    'historical_volatility': historical_volatility, 
    'current_price': current_price
})}
"""
        return analysis

    def _generate_scalping_insights(self, bollinger, stochastic, context):
        """Generate scalping insights based on indicators"""
        insights = []
        
        # Bollinger Band Analysis
        if context['current_price'] > bollinger['upper_band']:
            insights.append("🔴 Price Above Upper Band: Potential Overextension")
        elif context['current_price'] < bollinger['lower_band']:
            insights.append("🟢 Price Below Lower Band: Potential Reversal")
        else:
            insights.append("⚪ Price Within Bands: Neutral Trend")
        
        # Stochastic Oscillator Analysis
        if stochastic['k_line'] > 80:
            insights.append("⚠️ Stochastic Overbought: Consider Short Entry")
        elif stochastic['k_line'] < 20:
            insights.append("⚠️ Stochastic Oversold: Consider Long Entry")
        
        # Volatility Recommendation
        if context['historical_volatility'] > 2:
            insights.append("🔥 High Volatility: Tighter stop-losses recommended")
        else:
            insights.append("❄️ Low Volatility: Be cautious of false breakouts")
        
        return chr(10).join(insights)

def send_discord_message(webhook_url, message):
    """Send message to Discord webhook"""
    max_length = 2000
    for i in range(0, len(message), max_length):
        chunk = message[i:i+max_length]
        try:
            response = requests.post(webhook_url, json={"content": chunk})
            response.raise_for_status()
        except Exception as e:
            print(f"Error sending Discord message: {e}")

def main():
    # Discord webhook URL
    webhook_url = "https://discord.com/api/webhooks/1332276762603683862/aKE2i67QHm-1XR-HsMcQylaS0nKTS4yCVty4-jqvJscwkr6VRTacvLhP89F-4ABFDoQw"
    
    # Initialize scalping analysis
    scalping_analysis = ScalpingAnalysis()
    
    # Analyze ES Futures and S&P 500
    symbols = ['ES=F', '^GSPC']
    
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
