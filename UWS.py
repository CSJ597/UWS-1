import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
import seaborn as sns
import requests
from io import BytesIO
import base64
import datetime
import logging
from openai import OpenAI

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
        """Generate a comprehensive technical analysis chart"""
        plt.figure(figsize=(12, 8), facecolor="#a3c1ad")  # Set the facecolor here

        # Price and Bollinger Bands
        close_prices = data['Close']
        window = min(20, len(close_prices))  # Adjust window size if data is less
        
        middle_band = close_prices.rolling(window=window).mean()
        std_dev = close_prices.rolling(window=window).std()
        upper_band = middle_band + (std_dev * 2)
        lower_band = middle_band - (std_dev * 2)

        # Convert index to EST timezone
        est_index = data.index.tz_convert('US/Eastern')

        # Plot with white line for the price
        plt.plot(est_index, close_prices, label='Close Price', color='white', linewidth=1.5)
        plt.plot(est_index, upper_band, label='Upper Band', color='red', linestyle=':')
        plt.plot(est_index, lower_band, label='Lower Band', color='green', linestyle=':')

        plt.title('Underground Wall Street\nE-Mini S&P 500 TA', pad=20, color='white')
        plt.xlabel('Time (EST)', color='white')
        plt.ylabel('Price', color='white')
        plt.legend(facecolor='none', edgecolor='none', fontsize='small', loc='upper left')
        plt.grid(True, color='white')  # Set grid color to white

        # Format x-axis to display times in EST
        plt.xticks(rotation=45, ha='right')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S', tz=pytz.timezone('US/Eastern')))

        # Save plot to buffer with matching facecolor
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', facecolor="#a3c1ad", transparent=True)  # Ensure facecolor matches
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
        
        # Analyze the last 3 hours (180 minutes)
        last_data = data.tail(180)
        close_prices = last_data['Close']
        returns = close_prices.pct_change()

        # Bollinger Bands
        window = min(10, len(close_prices))  # 10-period Bollinger Bands for faster responsiveness.

        # Trend Analysis
        first_close = float(close_prices.iloc[0])
        last_close = float(close_prices.iloc[-1])
        price_range = float(last_data['High'].max() - last_data['Low'].min())
        cv = float((np.std(close_prices) / np.mean(close_prices)) * 100)  # Ensure cv is a scalar
        if cv < 0.3:
            trend = "RANGING"
        elif last_close > first_close and price_range > 0:
            trend = "BULLISH TREND"
        elif last_close < first_close and price_range > 0:
            trend = "BEARISH TREND"
        else:
            trend = "RANGING"

        # Volatility
        recent_returns = returns.tail(30)  # Use last 30 minutes for scalping volatility.
        volatility = float(np.std(recent_returns.dropna()) * np.sqrt(252) * 100)

        # Volume Sensitivity
        recent_volume = data['Volume'].tail(10).mean()
        avg_volume = data['Volume'].mean()
        volume_spike = recent_volume > (1.5 * avg_volume)

        # Compute metrics
        try:
            analysis = {
                'symbol': 'ES',  # Display as ES instead of ES=F
                'current_price': float(close_prices.iloc[-1]),
                'daily_change': float(returns.iloc[-1].item() * 100),
                'volatility': float(np.std(returns.dropna()) * np.sqrt(252) * 100),
                'market_trend': self.identify_market_trend(data),
                'technical_chart': self.generate_technical_chart(data, 'ES'),
                'session_high': float(last_data['High'].max()),
                'session_low': float(last_data['Low'].min()),
                'prev_close': prev_close,
                'volume': int(data['Volume'].sum()) if 'Volume' in data else None,
                'avg_volume': info.get('averageVolume', None),
                'description': info.get('shortName', 'E-mini S&P 500 Futures')
            }
        except Exception as e:
            return {'error': f'Analysis error: {str(e)}'}
        
        # Initialize OpenAI client with correct base URL
        api_key = '512dc9f0dfe54666b0d98ff42746dd13'
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.aimlapi.com"
        )

        # Prepare market data for analysis
        market_data = f"Current Price: {analysis['current_price']}, Daily Change: {analysis['daily_change']}, Volatility: {analysis['volatility']}, Market Trend: {analysis['market_trend']}, Session High: {analysis['session_high']}, Session Low: {analysis['session_low']}, Previous Close: {analysis['prev_close']}, Volume: {analysis['volume']}, Average Volume: {analysis['avg_volume']}"
        
        try:
            # Create chat completion with correct model name and format
            completion = client.chat.completions.create(
                model="deepseek-ai/deepseek-67b-chat",
                messages=[
                    {"role": "system", "content": "You are a professional market analyst. Analyze the given market data and provide detailed insights about the market trends, volatility, and potential outlook."},
                    {"role": "user", "content": f"Please analyze this market data and provide insights: {market_data}"}
                ],
                temperature=0.7,
                max_tokens=256
            )
            
            # Extract the response
            ai_analysis = completion.choices[0].message.content
            logging.info(f"API Response: {ai_analysis}")
        except Exception as e:
            ai_analysis = f'Failed to retrieve analysis: {str(e)}'
            logging.error(f"API Error: {str(e)}")

        # Store the AI analysis result
        analysis['ai_analysis'] = ai_analysis
        
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
• Volatility: {'⚠️ High' if volatility_status == 'HIGH' else '✅ Favorable' if volatility_status == 'MODERATE' else '⚡ Calm'} 
• AI Analysis: {analysis['ai_analysis']}
{'─' * 15}
"""
        
        # Add chart image if present
        chart = analysis.get('technical_chart', None)

    return report, chart


if __name__ == "__main__":
    analyzer = MarketAnalysis()
    analysis_results = analyzer.analyze_market()
    
    if 'error' in analysis_results:
        print(f"Error in analysis: {analysis_results['error']}")
    else:
        # Generate and send report
        report, chart = generate_market_report([analysis_results])
        send_discord_message(DISCORD_WEBHOOK_URL, report, chart)

    print("Analysis completed and report sent. Script will stop now.")
