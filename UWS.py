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
import openai
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration (Replace with your actual values)
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1332904597848723588/cmauevZsGfVQ5u4zo9AHepkBU3dxHXrRT1swWFc0EoJ2O9WgGJIam202DXhpYbEIZi7o"
API_KEY = "32760184b7ce475e942fde2344d49a68"
FINLIGHT_API_KEY = "sk_ec789eebf83e294eb0c841f331d2591e7881e39ca94c7d5dd02645a15bfc6e52"  # Add your Finlight API key here

# Target run time in Eastern Time (24-hour format)
RUN_HOUR = 08 #  PM
RUN_MINUTE = 56

def wait_until_next_run():
    """Wait until the next scheduled run time on weekdays"""
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # Create target time for today
    target = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    
    # If we've already passed today's run time, move to tomorrow
    if now >= target:
        target += timedelta(days=1)
        target = target.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    
    # Keep moving forward days until we hit a weekday (Monday = 0, Sunday = 6)
    while target.weekday() > 4:  # Skip Saturday (5) and Sunday (6)
        target += timedelta(days=1)
        target = target.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    
    # Calculate sleep duration
    sleep_seconds = (target - now).total_seconds()
    if sleep_seconds > 0:
        logging.info(f"Waiting until next run time: {target.strftime('%Y-%m-%d %I:%M %p %Z')}")
        time.sleep(sleep_seconds)

class MarketAnalysis:
    def __init__(self):
        """Initialize market analysis."""
        self.analysis_config = {
            'period': '1d',  
            'interval': '1m',  
        }
        self.allowed_symbols = ['ES=F']
        self.eastern_tz = pytz.timezone('US/Eastern')
        self.last_analysis_time = 0
        self.cached_analysis = None
        self.cache_duration = 300  # Cache for 5 minutes
    
    def fetch_market_data(self, symbol):
        """
        Fetch comprehensive market data with error handling and retries
        
        Args:
            symbol (str): Stock/futures symbol to analyze
        
        Returns:
            tuple: (market data DataFrame, error message or None)
        """
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Get data with proper error handling
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")
                
                if data.empty:
                    if attempt < max_retries - 1:
                        logging.warning(f"No data received, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    return None, "No market data available"
                
                # Calculate market metrics
                close_prices = data['Close']
                returns = close_prices.pct_change()
                
                market_data = {
                    'symbol': symbol.replace('=F', ''),  # Clean up futures symbol
                    'current_price': float(close_prices.iloc[-1]),
                    'prev_close': float(close_prices.iloc[-2]) if len(close_prices) > 1 else None,
                    'daily_change': float(((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]) * 100),
                    'volatility': float(np.std(returns.dropna()) * np.sqrt(252) * 100),
                    'market_trend': self.identify_market_trend(data),
                    'technical_chart': self.generate_technical_chart(data, 'ES'),
                    'session_high': float(data['High'].max()),
                    'session_low': float(data['Low'].min()),
                    'volume': int(data['Volume'].sum()) if 'Volume' in data.columns else None,
                    'data': data  # Store raw data for charts
                }
                
                return market_data, None
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Error fetching market data (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logging.error(f"Failed to fetch market data after {max_retries} attempts: {str(e)}")
                    return None, f"Error fetching market data: {str(e)}"
        
        return None, "Failed to fetch market data after all retries"

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
            ai_analysis = self.get_ai_analysis(analysis)
            
            # Update analysis with AI response
            analysis.update({
                'ai_analysis': f"\n{ai_analysis}\n"
            })
            
            # Format the analysis message
            analysis_message = f"""\n\n
🎯 **Market Analysis Report** 📊
            
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
                # Send header for news section
                self.send_discord_message(DISCORD_WEBHOOK_URL, "🔔 **Latest Market News** 🔔")
                
                # Create headers with API key
                headers = {
                    'accept': 'application/json',
                    'X-API-KEY': FINLIGHT_API_KEY
                }
                
                # Make multiple API calls with different market-related queries
                queries = [
                    'S&P 500',
                    'ES futures',
                    'SPY trading',
                    'market futures'
                ]
                
                all_articles = []
                seen_sources = set()
                filtered_articles = []
                backup_articles = []

                for query in queries:
                    try:
                        # Make direct API call for articles with extended model and all fields
                        url = f'https://api.finlight.me/v1/articles?query={query}&sort=publishedAt&order=DESC&limit=20&model=extended&fields=title,description,content,summary,sentiment,confidence,authors,topics,entities'
                        response = requests.get(url, headers=headers)
                        response.raise_for_status()
                        
                        data = response.json()
                        if isinstance(data, dict) and 'articles' in data:
                            # Log the query and number of articles found
                            logging.info(f"Query '{query}' returned {len(data['articles'])} articles")
                            # Log the first article to see its structure
                            if data['articles']:
                                logging.info(f"Sample article fields: {list(data['articles'][0].keys())}")
                            all_articles.extend(data['articles'])
                    except Exception as e:
                        logging.warning(f"Error fetching articles for query '{query}': {str(e)}")
                        continue

                # Filter articles
                for article in all_articles:
                    # Skip if not a dict or missing required fields
                    if not isinstance(article, dict):
                        continue
                        
                    # Check if article is relevant to S&P 500/ES
                    title = article.get('title', '').lower()
                    content = article.get('content', '')  # Try to get full content
                    summary = article.get('summary', '')  # Try to get summary
                    description = article.get('description', '')  # Try to get description
                    
                    # Combine all text fields for relevance check
                    all_content = f"{title} {content} {summary} {description}".lower()
                    
                    relevant_terms = ['s&p', 'sp500', 'spy', 'es futures', 'es-mini', 'stock market', 'market futures', 'dow', 'nasdaq', 'market summary']
                    if not any(term in all_content for term in relevant_terms):
                        continue

                    # Get source
                    source = article.get('source', '')
                    if not source:
                        continue

                    # If source is new, add to filtered articles
                    if source not in seen_sources:
                        seen_sources.add(source)
                        filtered_articles.append(article)
                    else:
                        # Keep as backup in case we don't have enough unique sources
                        backup_articles.append(article)

                # Sort both lists by date
                def get_article_date(article):
                    if not isinstance(article, dict):
                        return datetime.min
                    date_str = article.get('publishDate', '')
                    if not date_str:
                        return datetime.min
                    try:
                        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except ValueError:
                        try:
                            return datetime.fromisoformat(date_str)
                        except ValueError:
                            return datetime.min

                filtered_articles = sorted(filtered_articles, key=get_article_date, reverse=True)
                backup_articles = sorted(backup_articles, key=get_article_date, reverse=True)

                # If we don't have enough articles from unique sources, add from backup
                while len(filtered_articles) < 3 and backup_articles:
                    filtered_articles.append(backup_articles.pop(0))

                # Ensure we have exactly 3 articles
                filtered_articles = filtered_articles[:3]

                # Process each article
                news_articles = []
                for article in filtered_articles:
                    processed_article = self.process_article(article)
                    if processed_article:
                        news_articles.append({
                            'title': article.get('title', 'No Title'),
                            'description': processed_article,
                            'url': article.get('link', ''),
                            'color': 3447003,  # Blue
                            'fields': [
                                {
                                    'name': 'Source',
                                    'value': article.get('source', 'Unknown'),
                                    'inline': True
                                },
                                {
                                    'name': 'Published',
                                    'value': get_article_date(article).strftime('%I:%M %p EST') if get_article_date(article) != datetime.min else 'Unknown',
                                    'inline': True
                                }
                            ]
                        })

                # Send each article as an embed
                for article in news_articles:
                    self.send_discord_message(DISCORD_WEBHOOK_URL, "", news_articles=[article])

            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching articles: {str(e)}")
                if e.response is not None:
                    logging.error(f"Response content: {e.response.text}")
                self.send_discord_message(DISCORD_WEBHOOK_URL, "Unable to fetch articles at this time.")
            except Exception as e:
                logging.error(f"Error processing articles: {str(e)}")
                self.send_discord_message(DISCORD_WEBHOOK_URL, "Unable to process articles at this time.")

            logging.info("Discord messages sent successfully in order: analysis, chart, news")
            logging.info("Analysis complete")
            
        except Exception as e:
            logging.error(f"Market analysis error: {str(e)}")
            raise

    def prepare_market_prompt(self, data):
        """Prepare the prompt for AI analysis."""
        try:
            # Get market data from the input
            current_price = data.get('current_price', 0)
            prev_close = data.get('prev_close', 0)
            daily_change = data.get('daily_change', 0)
            market_trend = data.get('market_trend', '')
            
            # Create a concise prompt under 256 chars
            prompt = f"ES futures: ${current_price:.2f}, prev ${prev_close:.2f}, {daily_change:+.2f}%, trend: {market_trend}. Provide a brief market sentiment analysis."
            
            return prompt
            
        except Exception as e:
            logging.error(f"Error preparing market prompt: {str(e)}")
            return f"Analyze ES futures. Change: {data.get('daily_change', 0):.2f}%"

    def get_ai_analysis(self, data):
        """Get AI analysis of market data."""
        try:
            current_time = time.time()
            
            # Check if we have a cached analysis that's still valid
            if self.cached_analysis and (current_time - self.last_analysis_time) < self.cache_duration:
                logging.info("Using cached market analysis")
                return self.cached_analysis
            
            # Prepare the prompt with market data
            prompt = self.prepare_market_prompt(data)
            
            # Make API request with exponential backoff for rate limits
            max_retries = 3
            base_delay = 1  # Start with 1 second delay
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        'https://api.aimlapi.com/v1/chat/completions',
                        headers={'Authorization': f'Bearer {API_KEY}'},
                        json={
                            'model': 'gpt-3.5-turbo',
                            'messages': [{'role': 'user', 'content': prompt}],
                            'temperature': 0.7,
                            'max_tokens': 500
                        }
                    )
                    
                    # Log the request for debugging
                    logging.info(f"HTTP Request: POST https://api.aimlapi.com/v1/chat/completions \"{response.status_code} {response.reason}\"")
                    
                    if response.status_code == 429:  # Rate limit hit
                        if attempt < max_retries - 1:  # Don't sleep on the last attempt
                            delay = base_delay * (2 ** attempt)  # Exponential backoff
                            logging.info(f"Rate limit hit, waiting {delay} seconds before retry...")
                            time.sleep(delay)
                            continue
                    
                    response.raise_for_status()
                    analysis = response.json()['choices'][0]['message']['content']
                    
                    # Cache the successful analysis
                    self.cached_analysis = analysis
                    self.last_analysis_time = current_time
                    
                    return analysis
                    
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logging.info(f"Retrying request to /chat/completions in {delay} seconds")
                        time.sleep(delay)
                        continue
                    else:
                        logging.error(f"AI API Error: {str(e)}")
                        if hasattr(e, 'response') and e.response is not None:
                            logging.error(f"Error code: {e.response.status_code} - {e.response.text}")
                        
                        # Use fallback analysis and cache it
                        analysis = self.get_fallback_analysis(data)
                        self.cached_analysis = analysis
                        self.last_analysis_time = current_time
                        return analysis
            
            # If we've exhausted all retries, use fallback
            analysis = self.get_fallback_analysis(data)
            self.cached_analysis = analysis
            self.last_analysis_time = current_time
            return analysis
            
        except Exception as e:
            logging.error(f"Error in get_ai_analysis: {str(e)}")
            analysis = self.get_fallback_analysis(data)
            self.cached_analysis = analysis
            self.last_analysis_time = current_time
            return analysis

    def get_fallback_analysis(self, data):
        """Generate a fallback analysis when AI API is unavailable."""
        try:
            # Get market data
            current_price = data.get('current_price', 0)
            prev_close = data.get('prev_close', 0)
            daily_change = data.get('daily_change', 0)
            volatility = data.get('volatility', 0)
            market_trend = data.get('market_trend', '')
            session_high = data.get('session_high', 0)
            session_low = data.get('session_low', 0)
            
            # Determine trend based on available data
            if market_trend.lower() == 'bullish':
                trend = "bullish"
                strength = "showing strength"
            elif market_trend.lower() == 'bearish':
                trend = "bearish"
                strength = "showing weakness"
            else:
                trend = "neutral"
                strength = "consolidating"
            
            # Generate analysis based on available data
            analysis = [
                f"ES Futures Technical Analysis:",
                f"Current Price: {current_price:.2f}",
                f"Previous Close: {prev_close:.2f}",
                f"Daily Change: {daily_change:.2f}%",
                f"",
                f"Session Range:",
                f"- High: {session_high:.2f}",
                f"- Low: {session_low:.2f}",
                f"- Volatility: {volatility:.2f}%",
                f"",
                f"Market Trend: {trend.capitalize()}, {strength}",
            ]
            
            # Add volatility interpretation
            if volatility > 20:
                analysis.append("High volatility suggests increased market uncertainty")
            elif volatility < 10:
                analysis.append("Low volatility suggests market stability")
            
            # Add range analysis
            range_size = session_high - session_low
            avg_price = (session_high + session_low) / 2
            range_percent = (range_size / avg_price) * 100
            
            if range_percent > 1:
                analysis.append("Wide trading range indicates active price discovery")
            else:
                analysis.append("Narrow trading range suggests tight price consolidation")
            
            return "\n".join(analysis)
            
        except Exception as e:
            logging.error(f"Error in fallback analysis: {str(e)}")
            return f"Market Analysis: {'Up' if daily_change > 0 else 'Down'} {abs(daily_change):.2f}%"

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

    def process_article(self, article):
        """Process a single article and return its formatted content."""
        try:
            # Extract basic article info
            title = article.get('title', '')
            link = article.get('link', '')
            source = article.get('source', '')
            date = article.get('publishDate', '')
            sentiment = article.get('sentiment', '')
            confidence = article.get('confidence', '')
            
            # Try to get content from text field or fetch it
            content = None
            if link:
                try:
                    response = requests.get(link, timeout=5)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        # Get text content
                        text = soup.get_text()
                        # Clean up text
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        content = ' '.join(chunk for chunk in chunks if chunk)
                        if content:
                            content = content[:500] + "..."  # Limit length
                except Exception as e:
                    logging.error(f"Error fetching content: {str(e)}")
            
            # Build the article description
            content_parts = []
            
          
            
            # Add content if available
            if content:
                content_parts.append(f"📄 {content}")
            
            # Add sentiment if available
            if sentiment and confidence:
                content_parts.append(f"🎯 **Sentiment**: {sentiment.capitalize()} (Confidence: {float(confidence):.2%})")
            
            # Add source and date
            source_date = []
            if source:
                source_date.append(source)
            try:
                parsed_date = datetime.fromisoformat(date.replace('Z', '+00:00'))
                eastern_date = parsed_date.astimezone(self.eastern_tz)
                source_date.append(eastern_date.strftime('%I:%M %p ET'))
            except:
                if date:
                    source_date.append(date)
            
            if source_date:
                content_parts.append(f"*{' - '.join(source_date)}*")
            
          
            
            return '\n'.join(content_parts)
            
        except Exception as e:
            logging.error(f"Error processing article: {str(e)}")
            if title and link:
                return f"**{title}**\n[Read more]({link})"
            return None

    def fetch_article_content(self, link):
        """Fetch article content from the given link."""
        try:
            response = requests.get(link, timeout=5)
            if response.status_code == 200:
                # Extract main content using basic heuristics
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text and clean it up
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Take first 500 characters as content
                if text:
                    return text[:500] + "..."
        except Exception as e:
            logging.error(f"Error fetching article content: {str(e)}")
            return None

    def send_discord_message(self, webhook_url, content="", chart_base64=None, news_articles=None):
        """Send message to Discord with UWS branding"""
        
        # Define UWS branding
        username = "Underground Wall Street 🏦"
        avatar_url = "https://i.ibb.co/3N2NV0C/UWS-B-2.png"
        
        # Prepare the payload
        payload = {
            "username": username,
            "avatar_url": avatar_url
        }
        
        # Add content if provided
        if content:
            payload["content"] = "\n" + content
            
        # Add chart if provided
        if chart_base64:
            # Create a temporary file for the chart
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_file.write(base64.b64decode(chart_base64))
                temp_file.flush()
                
                # Send chart as file
                files = {
                    "file": ("chart.png", open(temp_file.name, "rb"), "image/png")
                }
                
                # Send with UWS branding
                response = requests.post(
                    webhook_url,
                    data={"username": username, "avatar_url": avatar_url},
                    files=files
                )
                response.raise_for_status()
                
                # Clean up temp file
                os.unlink(temp_file.name)
                return
        
        # Add news articles if provided
        if news_articles:
            payload["embeds"] = news_articles
        
        # Send the message
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()

def main():
    """Main function to run market analysis"""
    try:
        logging.info("Starting market analysis scheduler")
        
        while True:
            try:
                # Wait until next scheduled run time
                wait_until_next_run()
                
                # Create MarketAnalysis instance and run analysis
                analyzer = MarketAnalysis()
                analyzer.analyze_market()
                
                # Sleep for 30 seconds to avoid running multiple times
                time.sleep(30)
                
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                time.sleep(30)  # Wait 30 seconds before retrying
                
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()
