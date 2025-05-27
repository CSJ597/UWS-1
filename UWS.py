
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
RUN_HOUR = 22 #  1-24
RUN_MINUTE = 49 # 0-60

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

def calculate_rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    if len(close_prices) < window + 1:
        return pd.Series([np.nan] * len(close_prices), index=close_prices.index)
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method='bfill') # Backfill initial NaNs

def calculate_macd(close_prices: pd.Series, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Moving Average Convergence Divergence (MACD)."""
    if len(close_prices) < slow_window:
        nan_series = pd.Series([np.nan] * len(close_prices), index=close_prices.index)
        return nan_series, nan_series, nan_series
    exp1 = close_prices.ewm(span=fast_window, adjust=False).mean()
    exp2 = close_prices.ewm(span=slow_window, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.fillna(method='bfill'), signal_line.fillna(method='bfill'), histogram.fillna(method='bfill')

def calculate_bollinger_bands(close_prices: pd.Series, window: int = 20, num_std_dev: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    if len(close_prices) < window:
        nan_series = pd.Series([np.nan] * len(close_prices), index=close_prices.index)
        return nan_series, nan_series, nan_series
    rolling_mean = close_prices.rolling(window=window).mean()
    rolling_std = close_prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band.fillna(method='bfill'), rolling_mean.fillna(method='bfill'), lower_band.fillna(method='bfill')


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
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> dict:
        """
        Calculates RSI, MACD, and Bollinger Bands, adds them to the DataFrame,
        and returns their latest values.
        """
        close_prices = data['Close']
        
        # Calculate and add indicators to DataFrame
        data['RSI'] = calculate_rsi(close_prices)
        data['MACD_Line'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(close_prices)
        data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(close_prices)

        # Prepare dictionary of latest indicator values
        latest_indicators = {}
        if not data.empty:
            latest_indicators = {
                'rsi': round(data['RSI'].iloc[-1], 2) if pd.notna(data['RSI'].iloc[-1]) else None,
                'macd_line': round(data['MACD_Line'].iloc[-1], 2) if pd.notna(data['MACD_Line'].iloc[-1]) else None,
                'macd_signal': round(data['MACD_Signal'].iloc[-1], 2) if pd.notna(data['MACD_Signal'].iloc[-1]) else None,
                'macd_hist': round(data['MACD_Hist'].iloc[-1], 2) if pd.notna(data['MACD_Hist'].iloc[-1]) else None,
                'bb_upper': round(data['BB_Upper'].iloc[-1], 2) if pd.notna(data['BB_Upper'].iloc[-1]) else None,
                'bb_middle': round(data['BB_Middle'].iloc[-1], 2) if pd.notna(data['BB_Middle'].iloc[-1]) else None,
                'bb_lower': round(data['BB_Lower'].iloc[-1], 2) if pd.notna(data['BB_Lower'].iloc[-1]) else None,
            }
        return latest_indicators

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
        Identify market trend with more nuance (e.g., Strong/Weak Bullish/Bearish, Volatile/Quiet Ranging)
        
        Args:
            data (pd.DataFrame): Market price data with 'Close', 'High', 'Low' columns
        
        Returns:
            str: Market trend classification
        """
        if len(data) < 20:  # Increased minimum data points for SMA calculation
            return "INSUFFICIENT DATA"
        try:
            close_prices = data['Close']
            
            # Calculate price range and average price
            price_range = float(data['High'].max() - data['Low'].min())
            avg_price = float(close_prices.mean()) 
            
            if avg_price == 0:
                return "UNDEFINED (Zero Avg Price)"
            
            # Calculate standard deviation and Coefficient of Variation (CV)
            price_std = float(close_prices.std()) 
            cv = (price_std / avg_price) * 100  # Volatility percentage
            
            # Calculate a short-term SMA (e.g., 20-period)
            sma_short_period = 20 
            if len(close_prices) < sma_short_period:
                 return "INSUFFICIENT DATA FOR SMA" # Should be caught by initial check
            sma_short = close_prices.rolling(window=sma_short_period).mean()
            
            # Determine SMA slope (simple difference for recent trend)
            # Ensure there are enough points for slope calculation after dropping NaNs from rolling mean
            sma_dropna = sma_short.dropna()
            if len(sma_dropna) < 2: # Need at least two points to compare
                sma_slope_normalized = 0 # Neutral slope if not enough data
            else:
                # Compare the latest SMA value with an earlier SMA value (e.g., 10 periods ago from the available SMA series)
                # This gives a more robust slope than just last vs. second-to-last.
                comparison_point_index = max(0, len(sma_dropna) - (sma_short_period // 2) -1) # Look back about half the SMA window
                sma_slope = sma_dropna.iloc[-1] - sma_dropna.iloc[comparison_point_index]
                sma_slope_normalized = (sma_slope / avg_price) * 100 # Normalize slope by avg_price for consistent comparison

            # Trend classification logic
            # CV thresholds: Low < 0.3%, Moderate 0.3-1.0%, High > 1.0%
            # SMA Slope thresholds (normalized): Strong > 0.1%, Weak > 0.02%
            
            strong_slope_thresh_norm = 0.1 
            weak_slope_thresh_norm = 0.02

            if sma_slope_normalized > strong_slope_thresh_norm:
                if cv > 1.0: 
                    return "STRONG BULLISH (Volatile)"
                else: 
                    return "STRONG BULLISH (Steady)"
            elif sma_slope_normalized > weak_slope_thresh_norm:
                if cv > 1.0:
                     return "WEAK BULLISH (Volatile)"
                else:
                     return "WEAK BULLISH (Quiet)"
            elif sma_slope_normalized < -strong_slope_thresh_norm:
                if cv > 1.0:
                    return "STRONG BEARISH (Volatile)"
                else:
                    return "STRONG BEARISH (Steady)"
            elif sma_slope_normalized < -weak_slope_thresh_norm:
                if cv > 1.0:
                    return "WEAK BEARISH (Volatile)"
                else:
                    return "WEAK BEARISH (Quiet)"
            else: # SMA slope is relatively flat
                if cv > 1.0: 
                    return "VOLATILE RANGING"
                elif cv > 0.3: 
                    return "MODERATE RANGING"
                else: 
                    return "QUIET RANGING"
                
        except Exception as e:
            logging.error(f"Error in identify_market_trend: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return "TREND ANALYSIS ERROR"

    def generate_technical_chart(self, data, symbol):
        """Generate a comprehensive technical analysis chart"""
        # Ensure data has a proper index for plotting if it's not already DatetimeIndex
        # For this implementation, we'll assume data.index is suitable or convert to a simple range index for plotting.
        plot_df = data.reset_index()

        # Create figure with three subplots: Price+BB, RSI, MACD
        fig, (ax_price, ax_rsi, ax_macd) = plt.subplots(
            3, 1, figsize=(16, 12), sharex=True, 
            gridspec_kw={'height_ratios': [3, 1, 1]}
        )
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#1e222d')

        # --- Price and Bollinger Bands Subplot (ax_price) ---
        ax_price.set_facecolor('#1e222d')
        ax_price.plot(plot_df.index, plot_df['Close'], label='Price', color='cyan', linewidth=1.5)
        if 'BB_Upper' in plot_df.columns and 'BB_Middle' in plot_df.columns and 'BB_Lower' in plot_df.columns:
            ax_price.plot(plot_df.index, plot_df['BB_Upper'], label='Upper Band', color='dimgray', linestyle='--', linewidth=1)
            ax_price.plot(plot_df.index, plot_df['BB_Middle'], label='Middle Band (SMA20)', color='darkorange', linestyle='--', linewidth=1.2)
            ax_price.plot(plot_df.index, plot_df['BB_Lower'], label='Lower Band', color='dimgray', linestyle='--', linewidth=1)
            ax_price.fill_between(plot_df.index, plot_df['BB_Lower'], plot_df['BB_Upper'], alpha=0.1, color='darkorange')
        
        ax_price.set_title(f'UWS: {symbol} Analysis - Price & Bollinger Bands', color='white', pad=15)
        ax_price.set_ylabel('Price', color='white')
        ax_price.tick_params(axis='y', colors='white')
        ax_price.grid(True, alpha=0.2, color='gray')
        ax_price.legend(loc='upper left')
        # Add percentage change text
        if not plot_df.empty:
            pct_change = ((plot_df['Close'].iloc[-1] - plot_df['Close'].iloc[0]) / plot_df['Close'].iloc[0]) * 100
            ax_price.text(0.02, 0.95, f'{pct_change:+.2f}%',
                            transform=ax_price.transAxes, color='white',
                            fontsize=12, fontweight='bold', va='top')

        # --- RSI Subplot (ax_rsi) ---
        ax_rsi.set_facecolor('#1e222d')
        if 'RSI' in plot_df.columns:
            ax_rsi.plot(plot_df.index, plot_df['RSI'], label='RSI', color='magenta', linewidth=1.5)
            ax_rsi.axhline(70, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            ax_rsi.axhline(50, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
            ax_rsi.axhline(30, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
        ax_rsi.set_title('Relative Strength Index (RSI)', color='white', pad=10, fontsize=10)
        ax_rsi.set_ylabel('RSI', color='white')
        ax_rsi.set_ylim(0, 100)
        ax_rsi.tick_params(axis='y', colors='white')
        ax_rsi.grid(True, alpha=0.2, color='gray')
        ax_rsi.legend(loc='upper left')

        # --- MACD Subplot (ax_macd) ---
        ax_macd.set_facecolor('#1e222d')
        if 'MACD' in plot_df.columns and 'MACD_Signal' in plot_df.columns and 'MACD_Hist' in plot_df.columns:
            ax_macd.plot(plot_df.index, plot_df['MACD'], label='MACD', color='lime', linewidth=1.5)
            ax_macd.plot(plot_df.index, plot_df['MACD_Signal'], label='Signal Line', color='red', linestyle='--', linewidth=1.5)
            # Color MACD Histogram bars
            bar_colors = ['green' if val >= 0 else 'red' for val in plot_df['MACD_Hist']]
            ax_macd.bar(plot_df.index, plot_df['MACD_Hist'], label='Histogram', color=bar_colors, alpha=0.5, width=0.7)
            ax_macd.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax_macd.set_title('MACD', color='white', pad=10, fontsize=10)
        ax_macd.set_ylabel('MACD', color='white')
        ax_macd.tick_params(axis='y', colors='white')
        ax_macd.grid(True, alpha=0.2, color='gray')
        ax_macd.legend(loc='upper left')

        # X-axis formatting (applied to the last subplot due to sharex=True)
        # Use original datetime index for labels if available, otherwise use integer index
        if isinstance(data.index, pd.DatetimeIndex):
            times = data.index.strftime('%H:%M') # Format as HH:MM
            step = max(1, len(plot_df) // 7) # Show around 7-8 labels
            ax_macd.set_xticks(plot_df.index[::step])
            ax_macd.set_xticklabels(times[::step], rotation=45, ha='right')
        else:
            step = max(1, len(plot_df) // 7)
            ax_macd.set_xticks(plot_df.index[::step])
            ax_macd.set_xticklabels(plot_df.index[::step], rotation=45, ha='right')
        ax_macd.tick_params(axis='x', colors='white')
        ax_macd.set_xlabel('Time', color='white')

        plt.tight_layout(pad=1.5) # Add some padding
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#1e222d') # Increased dpi slightly
        plt.close(fig) # Ensure the specific figure is closed
        
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
            data = ticker.history(period=self.analysis_config['period'], interval=self.analysis_config['interval']) # Use config
            
            if data.empty:
                logging.error(f"No data available for {symbol}")
                self.send_discord_message(DISCORD_WEBHOOK_URL, f"⚠️ Market Analysis Error: No data available for {symbol}.")
                return {'error': 'No data available'}
            
            # Clean symbol for display
            symbol_cleaned = symbol.replace('=F', '').replace('ES', 'S&P 500 E-mini') # More descriptive

            # Calculate technical indicators and add them to 'data' DataFrame
            # The 'data' DataFrame is modified in place by this call
            technical_indicators = self._calculate_technical_indicators(data)

            close_prices = data['Close']
            returns = close_prices.pct_change().dropna() # Drop NaNs from returns
            
            analysis = {
                'symbol': symbol_cleaned,
                'current_price': round(close_prices.iloc[-1], 2),
                'daily_change': round(((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]) * 100, 2),
                'volatility': round(np.std(returns) * np.sqrt(252 * 390) * 100, 2) if not returns.empty else 0, # Adjusted for 1m interval (390 min in trading day)
                'market_trend': self.identify_market_trend(data.copy()), # Pass a copy to avoid issues if identify_market_trend modifies it
                'technical_chart': self.generate_technical_chart(data.copy(), symbol_cleaned), # Pass copy with indicators
                'session_high': round(data['High'].max(), 2),
                'session_low': round(data['Low'].min(), 2),
                'prev_close': round(close_prices.iloc[-2], 2) if len(close_prices) > 1 else None,
                'volume': int(data['Volume'].iloc[-1]) if 'Volume' in data.columns and not data['Volume'].empty else None,
                'news_events': news_events,
                **technical_indicators # Merge the latest indicator values
            }
            
            # Generate advanced prompt (will be updated later to include new indicators)
            ai_prompt_payload = self._generate_advanced_prompt(analysis, news_events, None) 
            
            # AI Analysis with enhanced prompt
            ai_analysis_text = self.get_ai_analysis(ai_prompt_payload) # Pass payload directly
            
            analysis.update({
                'ai_analysis': f"\n{ai_analysis_text}\n" if ai_analysis_text else "\nAI analysis currently unavailable.\n"
            })
            
            # Format the analysis message (will be updated later to include new indicators)
            analysis_message = f"""\n\n
🎯 **Market Analysis Report for {analysis['symbol']}** 📊
        
📈 **Price Action**
• Current: ${analysis['current_price']:.2f}
• Range: ${analysis['session_low']:.2f} - ${analysis['session_high']:.2f}
• Daily Change: {analysis['daily_change']:.2f}%
• Volume: {analysis.get('volume', 'N/A'):,}

📊 **Market Conditions**
• Trend: {analysis['market_trend']}
• Volatility: {analysis['volatility']:.1f}%
• Momentum (Daily Change): {abs(analysis['daily_change']):.1f}%

"""
            # Add technical indicators to message if available
            if analysis.get('rsi') is not None:
                analysis_message += f"• RSI (14): {analysis['rsi']:.2f}\n"
            if analysis.get('macd_line') is not None:
                analysis_message += f"• MACD ({analysis.get('macd_line', 'N/A'):.2f}, {analysis.get('macd_signal', 'N/A'):.2f}, {analysis.get('macd_hist', 'N/A'):.2f})\n"
            if analysis.get('bb_upper') is not None:
                analysis_message += f"• Bollinger Bands: ({analysis.get('bb_lower', 'N/A'):.2f} - {analysis.get('bb_middle', 'N/A'):.2f} - {analysis.get('bb_upper', 'N/A'):.2f})\n"

            analysis_message += f"""
🔍 **AI Analysis**
{analysis['ai_analysis']}
"""

            # Send the analysis message first
            self.send_discord_message(DISCORD_WEBHOOK_URL, analysis_message)

            # Send the chart next (will be updated later to show new indicators)
            if analysis.get('technical_chart'):
                self.send_discord_message(DISCORD_WEBHOOK_URL, "", chart_base64=analysis['technical_chart'])
            else:
                logging.warning("Technical chart was not generated.")


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
                        
                        data_news = response.json() # Renamed to avoid conflict with market data
                        if isinstance(data_news, dict) and 'articles' in data_news:
                            logging.info(f"Query '{query}' returned {len(data_news['articles'])} articles")
                            if data_news['articles']:
                                logging.info(f"Sample article fields: {list(data_news['articles'][0].keys())}")
                            all_articles.extend(data_news['articles'])
                    except Exception as e:
                        logging.warning(f"Error fetching articles for query '{query}': {str(e)}")
                        continue

                # Filter articles
                for article in all_articles:
                    if not isinstance(article, dict):
                        continue
                        
                    title = article.get('title', '').lower()
                    content = article.get('content', '') 
                    summary = article.get('summary', '')
                    description = article.get('description', '')
                    
                    is_relevant = False
                    search_terms = ['s&p 500', 'es futures', 'spy']
                    for term in search_terms:
                        if term in title or term in (content or '') or term in (summary or '') or term in (description or ''):
                            is_relevant = True
                            break
                    
                    if not is_relevant:
                        continue
                    
                    article_date_str = article.get('publishedAt')
                    if article_date_str:
                        article_date = self.get_article_date(article) # Assuming get_article_date is defined
                        if article_date and (datetime.now(pytz.utc) - article_date).days > 0:
                            continue
                    
                    source_name = article.get('source', {}).get('name')
                    if source_name and source_name not in seen_sources:
                        filtered_articles.append(article)
                        seen_sources.add(source_name)
                    elif not source_name:
                        backup_articles.append(article)
                
                if len(filtered_articles) < 5 and backup_articles:
                    needed = 5 - len(filtered_articles)
                    filtered_articles.extend(backup_articles[:needed])
                
                if filtered_articles:
                    discord_news_embeds = []
                    for article_data in filtered_articles[:5]:
                        processed_article_content = self.process_article(article_data) # Assuming process_article is defined
                        if processed_article_content:
                            discord_news_embeds.append({
                                "title": article_data.get('title', 'No Title'),
                                "description": processed_article_content,
                                "url": article_data.get('url', ''),
                                "color": 0x1E90FF  # Dodger Blue
                            })
                    
                    if discord_news_embeds:
                        self.send_discord_message(DISCORD_WEBHOOK_URL, news_articles=discord_news_embeds)
                    else:
                        self.send_discord_message(DISCORD_WEBHOOK_URL, "No relevant news articles found today.")
                else:
                    self.send_discord_message(DISCORD_WEBHOOK_URL, "No relevant news articles found today.")


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
            prev_close = data.get('previous_close', 0) 
            daily_change = data.get('daily_change', 0)
            volatility = data.get('volatility', 0)
            market_trend_str = data.get('trend', '') 
            session_high = data.get('session_high', 0)
            session_low = data.get('session_low', 0)
            
            # Determine trend based on available data
            trend_label = "neutral"
            strength_label = "consolidating"
            if market_trend_str: 
                if 'bullish' in market_trend_str.lower():
                    trend_label = "bullish"
                    strength_label = "showing strength"
                elif 'bearish' in market_trend_str.lower():
                    trend_label = "bearish"
                    strength_label = "showing weakness"
                elif 'ranging' in market_trend_str.lower():
                    trend_label = "ranging"
                    strength_label = "market is ranging"
            
            # Generate analysis based on available data
            analysis_lines = [
                f"ES Futures Technical Analysis (Fallback Mode):",
                f"Current Price: {current_price:.2f}",
                f"Previous Close: {prev_close:.2f}", 
                f"Daily Change: {daily_change:.2f}%",
                f"Session High: {session_high:.2f}",
                f"Session Low: {session_low:.2f}",
                f"Volatility: {volatility:.2f}%",
                f"Market Trend: {trend_label.capitalize()}, {strength_label}.",
            ]
            
            # Add volatility interpretation
            if isinstance(volatility, (int, float)):
                if volatility > 1.5: 
                    analysis_lines.append("Note: Volatility is relatively high.")
                elif volatility < 0.5: 
                    analysis_lines.append("Note: Volatility is relatively low.")

            return "\n".join(analysis_lines)

            
        except Exception as e:
            logging.error(f"Error in fallback analysis: {str(e)}")
            # Attempt to get daily_change from data if possible, otherwise provide a generic message
            try:
                daily_change_val = data.get('daily_change', 0.0) # Default to 0.0 if not found
                return f"Fallback Analysis Error. Market: {'Up' if daily_change_val > 0 else 'Down'} {abs(daily_change_val):.2f}%. Details: {str(e)}"
            except:
                return f"Fallback Analysis Error. Unable to determine market direction. Details: {str(e)}"

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
            # market_data is the input, previously referred to as analysis_results in planning
            prompt_str = f"""
Analyze the current market conditions for {market_data.get('symbol', 'N/A')} based on the following data:
Current Price: {market_data.get('current_price', 'N/A')}
Previous Close: {market_data.get('previous_close', 'N/A')}
Open: {market_data.get('open_price', 'N/A')}
Day's High: {market_data.get('session_high', 'N/A')}
Day's Low: {market_data.get('session_low', 'N/A')}
Volume: {market_data.get('volume', 'N/A')}
50-day SMA: {market_data.get('sma50', 'N/A')}
200-day SMA: {market_data.get('sma200', 'N/A')}
Market Trend: {market_data.get('trend', 'N/A')}
Volatility (Std Dev of Price Changes): {market_data.get('volatility', 'N/A')}
RSI (14-day): {market_data.get('RSI', 'N/A')}
MACD Line: {market_data.get('MACD', 'N/A')}
MACD Signal Line: {market_data.get('MACD_Signal', 'N/A')}
MACD Histogram: {market_data.get('MACD_Hist', 'N/A')}
Bollinger Bands (20-day, 2 std dev):
  Upper Band: {market_data.get('BB_Upper', 'N/A')}
  Middle Band (SMA20): {market_data.get('BB_Middle', 'N/A')}
  Lower Band: {market_data.get('BB_Lower', 'N/A')}
Key News Headlines: {market_data.get('news_summary', 'No relevant news found.')}

Provide a concise analysis (max 3-4 sentences) covering:
1. Current market sentiment (bullish, bearish, neutral) and its strength, considering the technical indicators.
2. Key support and resistance levels, potentially indicated by Bollinger Bands or SMAs.
3. Potential short-term outlook (e.g., likely to test support/resistance, consolidate, breakout) based on price action relative to indicators.
4. Mention any significant news impact if applicable.
Focus on actionable insights. Avoid generic statements. Be specific and integrate the provided technical indicators into your reasoning.
"""

            messages = [
                {
                    "role": "system",
                    "content": "You are a sophisticated market analyst. Based on the detailed market data provided, offer a concise and actionable analysis. Integrate all technical indicators and news sentiment into your assessment."
                },
                {
                    "role": "user",
                    "content": prompt_str
                }
            ]

            return {
                "model": "gpt-4-turbo", 
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 350
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
    # main() # Commented out for single test run
    try:
        print("Starting single test run of MarketAnalysis...")
        logging.info("Executing single test run of MarketAnalysis...")
        analyzer = MarketAnalysis()
        analyzer.analyze_market()
        print("Test run completed successfully.")
        logging.info("Test run of MarketAnalysis completed successfully.")
    except Exception as e:
        import traceback
        print(f"Error during test run: {str(e)}")
        logging.error(f"Error during test run: {str(e)}\n{traceback.format_exc()}")
