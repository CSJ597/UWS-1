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
from bs4 import BeautifulSoup
import pytz

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

    def get_high_impact_news(self):
        """Get high-impact news events from ForexFactory"""
        try:
            # Get current date in EST
            now = datetime.datetime.now(self.eastern_tz)
            url = f"https://www.forexfactory.com/calendar?day={now.strftime('%b.%d.%Y')}"
            
            # Fetch the page
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all news events
            events = []
            rows = soup.find_all('tr', class_='calendar__row')
            
            for row in rows:
                # Check if it's a high-impact US event
                impact = row.find('div', class_='calendar__impact-icon')
                if impact and 'high' in str(impact).lower():
                    country = row.find('td', class_='calendar__currency')
                    if country and 'USD' in str(country):
                        # Get event time
                        time_cell = row.find('td', class_='calendar__time')
                        if time_cell:
                            time_str = time_cell.text.strip()
                            if time_str:
                                try:
                                    # Parse the time
                                    if ':' in time_str:
                                        hour, minute = map(int, time_str.replace('am', '').replace('pm', '').split(':'))
                                        if 'pm' in time_str.lower() and hour != 12:
                                            hour += 12
                                        elif 'am' in time_str.lower() and hour == 12:
                                            hour = 0
                                            
                                        event_time = now.replace(hour=hour, minute=minute)
                                        
                                        # Only include events within next 2 hours
                                        time_diff = (event_time - now).total_seconds() / 3600
                                        if -0.5 <= time_diff <= 2:
                                            # Get event details
                                            title = row.find('span', class_='calendar__event-title')
                                            forecast = row.find('td', class_='calendar__forecast')
                                            previous = row.find('td', class_='calendar__previous')
                                            
                                            events.append({
                                                'time': event_time.strftime('%-I:%M %p ET'),
                                                'title': title.text.strip() if title else 'Unknown',
                                                'forecast': forecast.text.strip() if forecast else 'N/A',
                                                'previous': previous.text.strip() if previous else 'N/A'
                                            })
                                except Exception as e:
                                    logging.error(f"Error parsing event time: {e}")
                                    continue
            
            return events
            
        except Exception as e:
            logging.error(f"Error fetching news: {e}")
            return []
            
    def analyze_market(self, symbol='ES=F'):
        """Comprehensive market analysis"""
        try:
            # Get data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            info = ticker.info
            
            if data.empty:
                return {'error': 'No data available'}
            
            # Get upcoming high-impact news
            news_events = self.get_high_impact_news()
            
            # Calculate metrics and prepare analysis data
            analysis = self._prepare_analysis_data(data, info, news_events)
            
            # Format market data (shorter version)
            market_data = f"ES ${analysis['current_price']:.2f} ({analysis['daily_change']:.1f}%) | H: ${analysis['session_high']:.2f} L: ${analysis['session_low']:.2f} | {analysis['market_trend']}"
            
            # Add critical news only
            if news_events:
                next_event = news_events[0]
                market_data += f"\nNews: {next_event['time']} - {next_event['title']}"
            
            # Try AI analysis first
            try:
                payload = {
                    "model": "gpt-4",
                    "messages": [
                        {
                            "role": "system", 
                            "content": "Analyze ES price action and key levels. Consider market structure and upcoming news. No indicators."
                        },
                        {"role": "user", "content": market_data}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 256
                }
                
                response = requests.post(
                    'https://api.aimlapi.com/v1/chat/completions',
                    json=payload,
                    headers={'api-key': 'YOUR-API-KEY'},
                    timeout=10
                )
                
                logging.info(f"API Response Status Code: {response.status_code}")
                
                if response.status_code in [200, 201]:
                    analysis['analysis'] = response.json()['choices'][0]['message']['content']
                elif response.status_code == 429:  # Rate limit reached
                    logging.warning("API rate limit reached, using basic analysis")
                    analysis['analysis'] = self.generate_basic_analysis(analysis)
                else:
                    error_msg = response.json().get('message', 'Unknown error')
                    logging.error(f"API Error: {error_msg}")
                    analysis['analysis'] = self.generate_basic_analysis(analysis)
                    
            except Exception as e:
                logging.error(f"API request failed: {str(e)}")
                analysis['analysis'] = self.generate_basic_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            logging.error(f"Market analysis error: {str(e)}")
            return {'error': str(e)}
            
    def _prepare_analysis_data(self, data, info, news_events):
        """Prepare analysis data from market information"""
        close_prices = data['Close']
        returns = close_prices.pct_change()
        
        # Get scalar values
        high_val = data['High'].max()
        low_val = data['Low'].min()
        first_close = data['Close'].iloc[0]
        last_close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2] if len(data) > 1 else first_close
        
        # Calculate trend
        close_values = close_prices.values
        cv = float((np.std(close_values) / np.mean(close_values)) * 100)
        
        if cv < 0.3:
            trend = "RANGING"
        elif last_close > prev_close:
            trend = "BULLISH"
        else:
            trend = "BEARISH"
            
        # Calculate volatility
        recent_returns = returns.tail(30).dropna()
        if len(recent_returns) > 0:
            volatility = float(np.std(recent_returns) * np.sqrt(252) * 100)
        else:
            volatility = 0
        
        return {
            'symbol': 'ES',
            'current_price': last_close,
            'daily_change': ((last_close - first_close) / first_close) * 100,
            'market_trend': trend,
            'volatility': volatility,
            'technical_chart': self.generate_technical_chart(data, 'ES'),
            'session_high': high_val,
            'session_low': low_val,
            'prev_close': prev_close,
            'volume': int(data['Volume'].iloc[0]) if 'Volume' in data.columns else None,
            'avg_volume': info.get('averageVolume', None),
            'description': info.get('shortName', 'E-mini S&P 500 Futures'),
            'news_events': news_events
        }

    def generate_basic_analysis(self, analysis):
        """Generate basic analysis when API is rate limited"""
        current_price = analysis['current_price']
        prev_close = analysis['prev_close']
        high = analysis['session_high']
        low = analysis['session_low']
        daily_change = analysis['daily_change']
        trend = analysis['market_trend']
        
        # Calculate price position
        range_size = high - low
        if range_size > 0:
            price_position = (current_price - low) / range_size * 100
        else:
            price_position = 50
            
        # Generate basic analysis
        analysis_text = [
            f"ES Price Action Analysis:",
            f"Current: ${current_price:.2f} ({daily_change:+.1f}%)",
            f"Day Range: ${low:.2f} - ${high:.2f}",
            f"Position in Range: {price_position:.1f}%",
            f"Market Structure: {trend}",
        ]
        
        # Add momentum analysis
        if current_price > prev_close:
            analysis_text.append("Momentum: Positive - Price above previous close")
        elif current_price < prev_close:
            analysis_text.append("Momentum: Negative - Price below previous close")
        else:
            analysis_text.append("Momentum: Neutral - Price at previous close")
            
        # Add news warning if present
        if analysis.get('news_events'):
            next_event = analysis['news_events'][0]
            analysis_text.append(f"\nCAUTION: High Impact News at {next_event['time']}")
            analysis_text.append(f"Event: {next_event['title']}")
            
        return "\n".join(analysis_text)

    def send_discord_message(self, webhook_url, content, chart_data=None):
        """Send message to Discord with better formatting and error handling"""
        try:
            # Truncate content if too long
            max_content_length = 1500  # Discord's limit is 2000, leaving room for formatting
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."

            # Format the message
            formatted_content = (
                "```ansi\n"  # Use ansi for better formatting
                "🟦 ES MARKET ANALYSIS\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{content}\n"
                "```"
            )

            # Prepare the payload
            payload = {"content": formatted_content}
            files = {}

            # Add chart if available
            if chart_data:
                try:
                    # Decode base64 chart data
                    chart_bytes = base64.b64decode(chart_data)
                    files = {
                        'chart.png': ('chart.png', chart_bytes, 'image/png')
                    }
                except Exception as e:
                    logging.error(f"Error processing chart data: {e}")

            # Send the message
            response = requests.post(
                webhook_url,
                data=payload,
                files=files,
                timeout=10
            )

            # Check response
            if response.status_code in [200, 204]:
                logging.info("Discord message sent successfully")
            else:
                error_msg = f"Discord API error: {response.status_code} - {response.text}"
                logging.error(error_msg)
                # Retry with just text if file upload failed
                if files and response.status_code == 400:
                    logging.info("Retrying without chart...")
                    response = requests.post(
                        webhook_url,
                        json={"content": formatted_content},
                        timeout=10
                    )
                    if response.status_code in [200, 204]:
                        logging.info("Discord message sent successfully (without chart)")
                    else:
                        logging.error(f"Retry failed: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            logging.error(f"Network error sending Discord message: {e}")
        except Exception as e:
            logging.error(f"Unexpected error sending Discord message: {e}")

def generate_market_report(analyses):
    """Generate a comprehensive market report from analyses"""
    report_lines = []
    chart = None
    
    for analysis in analyses:
        if 'error' in analysis:
            report_lines.append(f"Error: {analysis['error']}")
            continue
            
        # Get the chart from the first valid analysis
        if not chart and 'technical_chart' in analysis:
            chart = analysis['technical_chart']
            
        # Format the analysis section
        price = analysis['current_price']
        change = analysis['daily_change']
        trend = analysis['market_trend']
        high = analysis['session_high']
        low = analysis['session_low']
        
        # Basic price information
        report_lines.extend([
            f"ES Analysis Report",
            f"Price: ${price:.2f} ({change:+.1f}%)",
            f"Range: ${low:.2f} - ${high:.2f}",
            f"Trend: {trend}",
            ""
        ])
        
        # Add volatility if available
        if 'volatility' in analysis:
            vol = analysis['volatility']
            vol_status = "LOW" if vol < 15 else "HIGH" if vol > 30 else "MODERATE"
            report_lines.append(f"Volatility: {vol_status} ({vol:.1f}%)")
            
        # Add news events if available
        if analysis.get('news_events'):
            report_lines.extend([
                "",
                "High Impact News Events:"
            ])
            for event in analysis['news_events']:
                report_lines.append(f"{event['time']} - {event['title']}")
                
        # Add AI or basic analysis
        if 'analysis' in analysis:
            report_lines.extend([
                "",
                "Analysis:",
                analysis['analysis']
            ])
            
    return "\n".join(report_lines), chart

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
