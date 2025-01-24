import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
from io import BytesIO
import base64
import datetime
import logging

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
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        plt.style.use('dark_background')
        
        # Plot full timeframe
        self._plot_ohlcv(data, ax1, f'{symbol} Full View')
        
        # Plot last hour
        last_hour_data = data.tail(60)  # Last 60 minutes
        self._plot_ohlcv(last_hour_data, ax2, f'{symbol} Last Hour')
        
        plt.tight_layout()
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Encode
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def _plot_ohlcv(self, data, ax, title):
        """Plot OHLCV data on the given axis"""
        # Reset index for plotting
        df = data.reset_index()
        df.index = range(len(df))
        
        # Plot candlesticks
        up = df['Close'] > df['Open']
        down = ~up
        
        # Plot up candlesticks
        for idx in df.index[up]:
            ax.bar(idx, df.loc[idx, 'High'] - df.loc[idx, 'Low'],
                  bottom=df.loc[idx, 'Low'], width=0.8, color='#26a69a', alpha=0.3)
            ax.bar(idx, df.loc[idx, 'Close'] - df.loc[idx, 'Open'],
                  bottom=df.loc[idx, 'Open'], width=0.8, color='#26a69a')
        
        # Plot down candlesticks
        for idx in df.index[down]:
            ax.bar(idx, df.loc[idx, 'High'] - df.loc[idx, 'Low'],
                  bottom=df.loc[idx, 'Low'], width=0.8, color='#ef5350', alpha=0.3)
            ax.bar(idx, df.loc[idx, 'Close'] - df.loc[idx, 'Open'],
                  bottom=df.loc[idx, 'Open'], width=0.8, color='#ef5350')
        
        # Format x-axis with original timestamps
        times = data.index.strftime('%I:%M %p').str.lstrip('0')
        ax.set_xticks(range(0, len(df), max(1, len(df)//5)))
        ax.set_xticklabels(times[::max(1, len(df)//5)], rotation=45)
        
        # Add volume at the bottom
        if 'Volume' in df.columns:
            volume_ax = ax.twinx()
            volume_ax.bar(df.index, df['Volume'],
                        alpha=0.3, color='gray', width=0.8)
            volume_ax.set_ylabel('Volume')
            volume_ax.tick_params(axis='y', labelcolor='gray')
        
        # Customize appearance
        ax.set_title(title)
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)

    def analyze_market(self, symbol='ES=F'):
        """Comprehensive market analysis"""
        try:
            # Get data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            info = ticker.info
            
            if data.empty:
                return {'error': 'No data available'}
            
            # Calculate metrics
            close_prices = data['Close']
            returns = close_prices.pct_change()
            last_data = data.tail(1)
            
            # Get scalar values to avoid FutureWarnings
            high_val = data['High'].iloc[0]
            low_val = data['Low'].iloc[0]
            price_range = high_val - low_val
            
            first_close = data['Close'].iloc[0]
            last_close = data['Close'].iloc[-1]
            
            # Calculate standard deviation and mean using numpy on values
            close_values = close_prices.values
            cv = float((np.std(close_values) / np.mean(close_values)) * 100)
            
            if cv < 0.3:
                trend = "RANGING"
            elif last_close > first_close and price_range > 0:
                trend = "BULLISH"
            else:
                trend = "BEARISH"
            
            # Volatility calculation using numpy on values
            recent_returns = returns.tail(30)
            returns_values = recent_returns.dropna().values
            volatility = float(np.std(returns_values) * np.sqrt(252) * 100)
            
            # Prepare analysis dictionary with scalar values
            analysis = {
                'symbol': 'ES',
                'current_price': last_close,
                'daily_change': ((last_close - first_close) / first_close) * 100,
                'volatility': volatility,
                'market_trend': trend,
                'technical_chart': self.generate_technical_chart(data, 'ES'),
                'session_high': high_val,
                'session_low': low_val,
                'prev_close': data['Close'].iloc[-2],
                'volume': int(data['Volume'].iloc[0]) if 'Volume' in data.columns else None,
                'avg_volume': info.get('averageVolume', None),
                'description': info.get('shortName', 'E-mini S&P 500 Futures')
            }
            
            # Prepare shorter market data format with spacing
            market_data = (
                f"\nES ${analysis['current_price']:.2f} ({analysis['daily_change']:.1f}%)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Volatility: {analysis['volatility']:.1f}%\n"
                f"Trend: {analysis['market_trend']}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            
            # Prepare the API request with shorter prompts
            payload = {
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a market analyst. Provide a brief but insightful analysis focusing on:\n1) Technical interpretation of the current market state\n2) Key factors driving the price action\n3) Short-term directional bias\nDo not repeat the numerical data provided."
                    },
                    {"role": "user", "content": market_data}
                ],
                "temperature": 0.7,
                "max_tokens": 256
            }
            
            try:
                # Make the API request
                response = requests.post(
                    'https://api.aimlapi.com/v1/chat/completions',
                    headers={
                        'Authorization': f'Bearer 512dc9f0dfe54666b0d98ff42746dd13',
                        'Content-Type': 'application/json'
                    },
                    json=payload
                )
                logging.info(f"API Response Status Code: {response.status_code}")
                
                # Handle successful responses (both 200 and 201)
                if response.status_code in [200, 201]:
                    result = response.json()
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                    # Format the analysis with clear sections
                    ai_analysis = f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\nMARKET ANALYSIS:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n{content}"
                    logging.info(f"Analysis received: {ai_analysis[:10000]}...")
                else:
                    ai_analysis = '\n\nFailed to retrieve analysis'
                    logging.error(f"API Error: {response.text}")
            except Exception as e:
                ai_analysis = f'\n\nFailed to retrieve analysis: {str(e)}'
                logging.error(f"API Error: {str(e)}")

            # Store the AI analysis result
            analysis['ai_analysis'] = ai_analysis
            
            return analysis

        except Exception as e:
            logging.error(f"Market analysis error: {str(e)}")
            return {'error': str(e)}

    def send_discord_message(self, webhook_url, message, chart_base64=None):
        """Send a message to Discord with optional chart image"""
        try:
            payload = {"content": message}
            
            if chart_base64:
                # Create image file from base64
                image_data = base64.b64decode(chart_base64)
                files = {
                    'file': ('chart.png', image_data, 'image/png')
                }
                response = requests.post(webhook_url, data=payload, files=files)
            else:
                response = requests.post(webhook_url, json=payload)
            
            response.raise_for_status()
            logging.info("Discord message sent successfully")
            
        except Exception as e:
            logging.error(f"Failed to send Discord message: {str(e)}")
            raise

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
        analyzer.send_discord_message(DISCORD_WEBHOOK_URL, report, chart)

    print("Analysis completed and report sent. Script will stop now.")
