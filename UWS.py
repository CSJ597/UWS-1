import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import pytz
import logging
import base64
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
import time
import finlight_client
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration (Replace with your actual values)
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1332276762603683862/aKE2i67QHm-1XR-HsMcQylaS0nKTS4yCVty4-jqvJscwkr6VRTacvLhP89F-4ABFDoQw"
API_KEY = "bbbdc8f307d44bd6bc90f9920926abb4"
FINLIGHT_API_KEY = "sk_ec789eebf83e294eb0c841f331d2591e7881e39ca94c7d5dd02645a15bfc6e52"  # Add your Finlight API key here

# Target run time in Eastern Time (24-hour format)
RUN_HOUR = 20 #  PM
RUN_MINUTE = 25

def wait_until_next_run():
    """Wait until the next scheduled run time on weekdays"""
    et_tz = pytz.timezone('US/Eastern')
    now = datetime.now(et_tz)
    
    # Set target time using the configured hour and minute
    target = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    
    # If we're past today's run time, move to next day
    if now > target:
        target += timedelta(days=1)
    
    # Keep moving forward days until we hit a weekday (Monday = 0, Sunday = 6)
    while target.weekday() > 4:  # Skip Saturday (5) and Sunday (6)
        target += timedelta(days=1)
    
    # Calculate sleep duration
    sleep_seconds = (target - now).total_seconds()
    if sleep_seconds > 0:
        next_run = target.strftime('%Y-%m-%d %I:%M %p ET')
        logging.info(f"Waiting until next run at {next_run}")
        time.sleep(sleep_seconds)

class MarketAnalysis:
    def __init__(self):
        """Initialize analysis with default configurations"""
        self.analysis_config = {
            'period': '1d',  
            'interval': '1m',  
        }
        self.allowed_symbols = ['ES=F']
        self.eastern_tz = pytz.timezone('US/Eastern')
    
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
            
            if cv < 1.5:
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
        
        # Set the background color
        fig.patch.set_facecolor('#1e222d')
        ax1.set_facecolor('#1e222d')
        ax2.set_facecolor('#1e222d')
        
        # Plot full timeframe
        self._plot_ohlcv(data, ax1, f'UWS: {symbol} Full View')
        
        # Plot last hour
        last_hour_data = data.tail(60)  # Last 60 minutes
        self._plot_ohlcv(last_hour_data, ax2, f'UWS: {symbol} Last Hour')
        
        plt.tight_layout()
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                   facecolor='#1e222d', edgecolor='none')
        plt.close()
        
        # Encode
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def _plot_ohlcv(self, data, ax, title):
        """Plot OHLCV data on the given axis"""
        # Reset index for plotting
        df = data.reset_index()
        df.index = range(len(df))
        
        # Plot price line with gradient fill
        ax.plot(df.index, df['Close'], 
               color='white', linewidth=2, label='Price')
        
        # Add gradient fill
        ax.fill_between(df.index, df['Close'], df['Close'].min(),
                       alpha=0.1, color='white')
        
        # Add volume at the bottom
        if 'Volume' in df.columns:
            volume_ax = ax.twinx()
            volume_ax.fill_between(df.index, df['Volume'],
                               alpha=0.15, color='gray')
            volume_ax.set_ylabel('Volume', color='gray')
            volume_ax.tick_params(axis='y', colors='gray')
            volume_ax.grid(False)
        
        # Format x-axis with original timestamps
        times = data.index.strftime('%I:%M %p').str.lstrip('0')
        ax.set_xticks(range(0, len(df), max(1, len(df)//5)))
        ax.set_xticklabels(times[::max(1, len(df)//5)], rotation=45)
        
        # Customize appearance
        ax.set_title(title, color='white', pad=20)
        ax.set_ylabel('Price', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='gray')
        
        # Add percentage change
        pct_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
        ax.text(0.02, 0.95, f'{pct_change:+.2f}%', 
                transform=ax.transAxes, color='white',
                fontsize=12, fontweight='bold')
        logging.info(f'Plotting OHLCV data for {title}. Percentage change: {pct_change:.2f}%')

    def check_high_impact_news(self):
        """Check ForexFactory for high-impact news events"""
        try:
            # Get current time in ET
            now = datetime.now(self.eastern_tz)
            
            # Format today's date for URL
            date_str = now.strftime('%Y%m%d')
            url = f'https://www.forexfactory.com/calendar?day={date_str}'
            
            # Get the page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all news events
            events = soup.find_all('tr', class_='calendar__row')
            
            high_impact_events = []
            for event in events:
                # Check if it's a high impact event (red or orange folder)
                impact = event.find('div', class_='calendar__impact-icon')
                if not impact:
                    continue
                    
                impact_class = impact.get('class', [])
                if not any('high' in c or 'medium' in c for c in impact_class):
                    continue
                
                # Get event time
                time_elem = event.find('td', class_='calendar__time')
                if not time_elem:
                    continue
                    
                time_str = time_elem.text.strip()
                if not time_str or time_str == "All Day":
                    continue
                
                # Parse event time
                try:
                    event_time = datetime.strptime(time_str, '%I:%M%p').time()
                    event_datetime = datetime.combine(now.date(), event_time)
                    event_datetime = self.eastern_tz.localize(event_datetime)
                    
                    # Only include events within next 30 minutes
                    time_until_event = (event_datetime - now).total_seconds() / 60
                    if 0 <= time_until_event <= 1440:
                        # Get event details
                        currency = event.find('td', class_='calendar__currency').text.strip()
                        title = event.find('span', class_='calendar__event-title').text.strip()
                        
                        impact_level = 'High' if 'high' in ' '.join(impact_class) else 'Medium'
                        
                        high_impact_events.append({
                            'time': event_time.strftime('%I:%M %p'),
                            'currency': currency,
                            'event': title,
                            'impact': impact_level,
                            'minutes_until': int(time_until_event)
                        })
                except ValueError:
                    continue
            
            return high_impact_events
            
        except Exception as e:
            logging.error(f"Error checking news: {str(e)}")
            return []

    def analyze_market(self, symbol='ES=F'):
        """Comprehensive market analysis with enhanced AI prompt"""
        try:
            # Get high-impact news events
            news_events = self.check_high_impact_news()
            
            # Get data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if data.empty:
                return {'error': 'No data available'}
            
            close_prices = data['Close']
            returns = close_prices.pct_change()
            
            analysis = {
                'symbol': 'ES',
                'current_price': close_prices.iloc[-1],
                'daily_change': ((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]) * 100,
                'volatility': float(np.std(returns.dropna().values) * np.sqrt(252) * 100),
                'market_trend': self.identify_market_trend(data),
                'technical_chart': self.generate_technical_chart(data, 'ES'),
                'session_high': data['High'].iloc[0],
                'session_low': data['Low'].iloc[0],
                'prev_close': data['Close'].iloc[-2] if len(data) > 1 else None,
                'volume': int(data['Volume'].iloc[0]) if 'Volume' in data.columns else None,
                'news_events': news_events
            }
            
            # Generate advanced prompt
            ai_prompt = self._generate_advanced_prompt(analysis, news_events, None)
            
            # AI Analysis with enhanced prompt
            ai_analysis = ""
            try:
                if ai_prompt:
                    response = requests.post(
                        'https://api.aimlapi.com/v1/chat/completions',
                        json=ai_prompt,
                        headers={'Authorization': f'Bearer {API_KEY}'}
                    )
                    
                    if response.status_code in [200, 201]:
                        ai_analysis = response.json()['choices'][0]['message']['content'].strip()
                    else:
                        ai_analysis = "Unable to generate AI analysis. API returned an error."
                        logging.error(f"API Error: {response.text}")
                
            except Exception as e:
                ai_analysis = f"Analysis Error: {str(e)}"
                logging.error(f"Error getting AI analysis: {str(e)}")
            
            # Update analysis with AI response
            analysis.update({
                'ai_analysis': f"\n{ai_analysis}\n"
            })
            
            # Format the analysis message
            analysis_message = f"""🎯 **Market Analysis Report** 📊
            
📈 **Price Action**
• Current: ${analysis['current_price']:.2f}
• Range: ${analysis['session_low']:.2f} - ${analysis['session_high']:.2f}
• Daily Change: {analysis['daily_change']:.2f}%

📊 **Market Conditions**
• Trend: {analysis['market_trend']}
• Volatility: {analysis['volatility']:.1f}%
• Momentum: {abs(analysis['daily_change']):.1f}%

🔍 **AI Analysis**
{analysis['ai_analysis']}
"""

            # Send the analysis message first
            self.send_discord_message(DISCORD_WEBHOOK_URL, analysis_message)

            # Send the chart next
            self.send_discord_message(DISCORD_WEBHOOK_URL, "", chart_base64=analysis['technical_chart'])

            # Get articles from Finlight API
            try:
                client = finlight_client.FinlightApi({ 'apiKey': FINLIGHT_API_KEY })
                articles = []
                
                # Send header for news section
                self.send_discord_message(DISCORD_WEBHOOK_URL, "🔔 **Latest Market News** 🔔")
                
                # Get articles for each search query
                search_queries = ['S&P 500', 'ES futures', 'SPX', 'SPY ETF']
                for query in search_queries:
                    try:
                        response = client.articles.get_extended_articles({
                            'params': {
                                'query': query,
                                'language': "en"
                            }
                        })
                        
                        if isinstance(response, list):
                            articles.extend(response)
                        elif isinstance(response, dict):
                            if 'articles' in response:
                                articles.extend(response['articles'])
                            elif 'data' in response:
                                articles.extend(response['data'])
                            elif 'results' in response:
                                articles.extend(response['results'])
                    except Exception as e:
                        logging.error(f"Error searching for query '{query}': {str(e)}")
                        continue

                # Remove duplicates and sort by date
                unique_articles = []
                seen_urls = set()
                for article in articles:
                    url = article.get('url') or article.get('link')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        unique_articles.append(article)

                articles = sorted(
                    unique_articles,
                    key=lambda x: datetime.fromisoformat(x.get('publishedAt', '').replace('Z', '+00:00')),
                    reverse=True
                )[:3]  # Get top 3 articles

                # Send each article as an embed
                for article in articles:
                    embed = {
                        'title': article.get('title', 'No Title'),
                        'description': article.get('summary', article.get('description', 'No summary available')),
                        'url': article.get('url', article.get('link', '')),
                        'color': 3447003,  # Blue
                        'fields': [
                            {
                                'name': 'Source',
                                'value': article.get('source', article.get('publisher', 'Unknown')),
                                'inline': True
                            },
                            {
                                'name': 'Published',
                                'value': datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00')).strftime('%I:%M %p EST'),
                                'inline': True
                            }
                        ]
                    }
                    self.send_discord_message(DISCORD_WEBHOOK_URL, "", news_articles=[embed])

                logging.info("Discord messages sent successfully in order: analysis, chart, news")
                
            except Exception as e:
                logging.error(f"Error getting Finlight articles: {str(e)}")
                logging.error(f"Full error: {repr(e)}")
                # Send error message to Discord
                self.send_discord_message(DISCORD_WEBHOOK_URL, f"⚠️ Error fetching news articles: {str(e)}")

            logging.info("Analysis complete")
            
        except Exception as e:
            logging.error(f"Market analysis error: {str(e)}")
            raise

    def _generate_advanced_prompt(self, market_data, news_events, market_news):
        """
        Generate a concise market analysis prompt focusing on chart analysis
        
        Args:
            market_data (dict): Comprehensive market data
            news_events (list): Upcoming high-impact news events
            market_news (list): Recent market headlines (not used)
        
        Returns:
            dict: Structured prompt payload for AI analysis
        """
        try:
            # Prepare concise technical context
            technical_context = (f"Price: ${market_data['current_price']:.2f}, "
                                 f"Change: {market_data['daily_change']:.2f}%, "
                                 f"Trend: {market_data['market_trend']}, "
                                 f"Volatility: {market_data['volatility']:.2f}%")

            # Construct the prompt
            prompt_content = (
                f"Analyze the chart data:\n{technical_context}\n\n"
                "Provide:\n- Insights\n- Forecast"
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are a market analyst. Provide a detailed analysis based on the chart data."
                },
                {
                    "role": "user",
                    "content": prompt_content
                }
            ]

            return {
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 300  # Allow for detailed response
            }
        
        except Exception as e:
            logging.error(f"Error generating prompt: {str(e)}")
            return None

    def send_discord_message(self, webhook_url, message, chart_base64=None, avatar_url=None, news_articles=None):
        """Send a message to Discord with optional chart image and news articles"""
        try:
            # Prepare the base payload
            payload = {
                'username': 'Underground Wall Street 🏦',
                'avatar_url': avatar_url or 'https://i.ibb.co/3N2NV0C/UWS-B-2.png'
            }

            # Add message if provided
            if message:
                payload['content'] = message

            # Add embeds if provided
            if news_articles:
                payload['embeds'] = news_articles

            # Send message with embeds if present
            if message or news_articles:
                response = requests.post(webhook_url, json=payload)
                response.raise_for_status()
                time.sleep(1)  # Add delay between messages

            # Send chart as a separate message if provided
            if chart_base64:
                chart_bytes = base64.b64decode(chart_base64)
                files = {
                    'file': ('chart.png', chart_bytes, 'image/png')
                }
                chart_response = requests.post(webhook_url, files=files)
                chart_response.raise_for_status()
                time.sleep(1)  # Add delay between messages

        except Exception as e:
            logging.error(f"Error sending Discord message: {str(e)}")
            raise

def main():
    """Main function to run market analysis"""
    try:
        logging.info("Starting market analysis")
        market = MarketAnalysis()
        
        # Run analysis for ES futures
        market.analyze_market()
        
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()
