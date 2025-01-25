import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
from io import BytesIO
import base64
from datetime import datetime, timedelta
import logging
import pytz
from bs4 import BeautifulSoup
import re
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration (Replace with your actual values)
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1332276762603683862/aKE2i67QHm-1XR-HsMcQylaS0nKTS4yCVty4-jqvJscwkr6VRTacvLhP89F-4ABFDoQw"
API_KEY = "bbbdc8f307d44bd6bc90f9920926abb4"

# Target run time in Eastern Time (24-hour format)
RUN_HOUR = 14  #  PM
RUN_MINUTE = 00

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
    while target.weekday() > 7:  # Skip Saturday (5) and Sunday (6)
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
                    if 0 <= time_until_event <= 30:
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
            
    def get_marketwatch_news(self):
        """Scrape recent news from MarketWatch"""
        try:
            url = 'https://www.marketwatch.com/markets'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            news_items = []
            article_elements = soup.find_all('div', class_='article__content')
            
            for article in article_elements[:5]:  # Get latest 5 news items
                title_elem = article.find('h3', class_='article__headline')
                snippet_elem = article.find('p', class_='article__summary')
                time_elem = article.find('span', class_='article__timestamp')
                
                if title_elem and snippet_elem and time_elem:
                    title = title_elem.text.strip()
                    snippet = snippet_elem.text.strip()
                    timestamp = time_elem.text.strip()
                    
                    news_items.append({
                        'title': title,
                        'snippet': snippet,
                        'time': timestamp
                    })
            
            return news_items
            
        except Exception as e:
            logging.error(f"Error fetching MarketWatch news: {str(e)}")
            return []

    def _generate_advanced_prompt(self, market_data, news_events, market_news):
        """
        Generate a concise market analysis prompt focusing on chart analysis
        
        Args:
            market_data (dict): Comprehensive market data
            news_events (list): Upcoming high-impact news events (not used)
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
                "Provide:\n- Insights\n- Forecast\n- Suggested positions."
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

    def analyze_market(self, symbol='ES=F'):
        """Comprehensive market analysis with enhanced AI prompt"""
        try:
            # Existing data gathering steps remain the same
            news_events = self.check_high_impact_news()
            market_news = self.get_marketwatch_news()
            
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
                'news_events': news_events,
                'market_news': market_news
            }
            
            # Generate advanced prompt
            ai_prompt = self._generate_advanced_prompt(analysis, news_events, market_news)
            
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
            
            return analysis
            
        except Exception as e:
            logging.error(f"Market analysis error: {str(e)}")
            raise

    def send_discord_message(self, webhook_url, message, chart_base64=None):
        """Send a message to Discord with optional chart image"""
        try:
            # Prepare the message data
            data = {
                "content": message[:1900] + "..." if len(message) > 1900 else message,  # Discord has a 2000 char limit
                "username": "UWS Market Analysis"
            }
            
            files = {}
            
            # If we have a chart, add it as a file
            if chart_base64:
                try:
                    # Decode base64 string to bytes
                    chart_bytes = base64.b64decode(chart_base64)
                    
                    # Create file object
                    files = {
                        'file': ('chart.png', chart_bytes, 'image/png')
                    }
                except Exception as e:
                    logging.error(f"Error processing chart image: {str(e)}")
            
            # Send the message
            if files:
                response = requests.post(webhook_url, data=data, files=files)
            else:
                response = requests.post(webhook_url, json=data)
            
            response.raise_for_status()
            logging.info("Discord message sent successfully")
            
        except Exception as e:
            logging.error(f"Failed to send Discord message: {str(e)}")
            raise

    def generate_market_report(self, analysis_results):
        """Generate a comprehensive market report"""
        try:
            # Get the current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Create the report header
            report = f"MARKET ANALYSIS: {current_date}\n{'─' * 30}\n\n"
            
            chart = None
            
            for analysis in analysis_results:
                if 'error' in analysis:
                    report += f"Error: {analysis['error']}\n\n"
                    continue
                
                # Trading Insights
                volatility_status = "LOW" if analysis['volatility'] < 15 else "HIGH" if analysis['volatility'] > 30 else "MODERATE"
                
                # Calculate price position relative to day's range
                price_position = (analysis['current_price'] - analysis['session_low']) / (analysis['session_high'] - analysis['session_low']) * 100 if analysis['session_high'] != analysis['session_low'] else 50
                
                # Format price position description
                range_position = "NEAR HIGH" if price_position > 75 else "NEAR LOW" if price_position < 25 else "MID-RANGE"
                
                # Calculate change from previous close if available
                prev_close_info = ""
                if analysis['prev_close']:
                    change_from_prev = ((analysis['current_price'] - analysis['prev_close']) / analysis['prev_close']) * 100
                    prev_close_info = f"From Previous Close: {change_from_prev:+.2f}%\n"
                
                # Format news events
                news_section = "\nUPCOMING NEWS\n"
                if analysis.get('news_events'):
                    for event in sorted(analysis['news_events'], key=lambda x: x['minutes_until']):
                        impact_emoji = "🚨" if event['impact'] == "High" else "⚠️"
                        news_section += f"• {impact_emoji} {event['currency']} {event['event']} at {event['time']} ({event['minutes_until']}m)\n"
                else:
                    news_section += "• No high-impact news events in next 30 minutes\n"
                
                # Add recent market headlines
                if analysis.get('market_news'):
                    news_section += "\nRECENT HEADLINES\n"
                    for news in analysis['market_news'][:3]:
                        news_section += f"• {news['title']} ({news['time']})\n"
                
                report += f"""
📊 PRICE ACTION
• Current: ${analysis['current_price']:.2f} ({range_position})
• Range: ${analysis['session_low']:.2f} - ${analysis['session_high']:.2f}
{prev_close_info}
• Daily Change: {analysis['daily_change']:.2f}%

📈 MARKET CONDITIONS
• Trend: {analysis['market_trend']}
• Volatility: {volatility_status}
• Momentum: {abs(analysis['daily_change']):.1f}%

{news_section}

🔍 ANALYSIS
{analysis['ai_analysis']}

{'─' * 40}
"""
                
                # Add chart image if present
                chart = analysis.get('technical_chart', None)
            
            return report, chart
            
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")
            return f"Error generating report: {str(e)}", None

def main():
    """Main execution function for market analysis"""
    market = MarketAnalysis()
    
    while True:
        try:
            # Wait until next scheduled run
            wait_until_next_run()
            
            # Log start time
            start_time = datetime.now(pytz.timezone('US/Eastern'))
            logging.info(f"Starting market analysis at {start_time}")
            
            # Run analysis
            analysis_results = [market.analyze_market()]
            
            # Generate report
            report, chart = market.generate_market_report(analysis_results)
            
            # Send to Discord
            market.send_discord_message(DISCORD_WEBHOOK_URL, report, chart)
            
            logging.info("Analysis complete")
            
        except Exception as e:
            logging.error(f"Market analysis error: {str(e)}")
            time.sleep(60)  # Wait a minute before retrying on error

if __name__ == "__main__":
    main()
