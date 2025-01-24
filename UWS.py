import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
import base64
import datetime
import logging
import pytz

# Set up logging
logging.basicConfig(level=logging.INFO)

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1332276762603683862/aKE2i67QHm-1XR-HsMcQylaS0nKTS4yCVty4-jqvJscwkr6VRTacvLhP89F-4ABFDoQw"

class ScalpingAnalyzer:
    def __init__(self):
        self.config = {
            'interval': '1m',
            'rth_start': datetime.time(9, 30),
            'rth_end': datetime.time(16, 15),
            'bollinger_period': 10,
            'bollinger_std': 1.5,
            'rsi_period': 14,
            'volume_spike_window': (5, 30)
        }
        self.symbol = 'ES=F'
        self.tz = pytz.timezone('US/Eastern')

    def fetch_market_data(self):
        """Fetch and filter Regular Trading Hours data"""
        try:
            data = yf.download(
                self.symbol,
                period='1d',
                interval=self.config['interval'],
                prepost=False
            ).tz_convert(self.tz)
            
            # Filter RTH data
            rth_mask = (data.index.time >= self.config['rth_start']) & \
                      (data.index.time <= self.config['rth_end'])
            data_rth = data[rth_mask]
            
            if data_rth.empty:
                return data, "No RTH data available, using full session"
            
            return data_rth, None
        
        except Exception as e:
            return None, f"Data fetch error: {str(e)}"

    def calculate_indicators(self, data):
        """Calculate technical indicators for scalping"""
        if len(data) < 20:
            return {}, "Insufficient data for indicators"
            
        try:
            # Bollinger Bands
            close = data['Close']
            ma = close.rolling(self.config['bollinger_period']).mean()
            std = close.rolling(self.config['bollinger_period']).std()
            upper = ma + std * self.config['bollinger_std']
            lower = ma - std * self.config['bollinger_std']
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(alpha=1/self.config['rsi_period'], adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/self.config['rsi_period'], adjust=False).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # VWAP
            tp = (data['High'] + data['Low'] + data['Close']) / 3
            cumulative_tpv = (tp * data['Volume']).cumsum()
            cumulative_vol = data['Volume'].cumsum()
            vwap = cumulative_tpv / cumulative_vol
            
            return {
                'upper_band': upper,
                'lower_band': lower,
                'rsi': rsi,
                'vwap': vwap,
                'ma': ma
            }, None
        
        except Exception as e:
            return {}, f"Indicator error: {str(e)}"

    def analyze_price_action(self, data, indicators):
        """Generate scalping signals"""
        if data.empty:
            return {}
            
        current_price = data['Close'].iloc[-1]
        vwap = indicators['vwap'].iloc[-1] if 'vwap' in indicators else None
        rsi = indicators['rsi'].iloc[-1] if 'rsi' in indicators else None
        
        signals = {
            'price_vwap': 'Above' if current_price > vwap else 'Below' if vwap else None,
            'rsi_status': self._get_rsi_status(rsi),
            'bollinger_squeeze': self._detect_squeeze(indicators),
            'volume_spike': self._detect_volume_spike(data),
            'trend': self._detect_trend(data, indicators)
        }
        
        return signals

    def _get_rsi_status(self, rsi):
        if not rsi:
            return None
        if rsi > 70:
            return 'Overbought'
        if rsi < 30:
            return 'Oversold'
        return 'Neutral'

    def _detect_squeeze(self, indicators):
        if 'upper_band' not in indicators:
            return False
        recent_band_width = (indicators['upper_band'].iloc[-5:] - 
                            indicators['lower_band'].iloc[-5:]).mean()
        return recent_band_width < np.mean(indicators['upper_band'] - indicators['lower_band']) * 0.5

    def _detect_volume_spike(self, data):
        vol_window, vol_avg_window = self.config['volume_spike_window']
        recent_vol = data['Volume'].iloc[-vol_window:].mean()
        avg_vol = data['Volume'].iloc[-vol_avg_window:].mean()
        return recent_vol > avg_vol * 2 if avg_vol > 0 else False

    def _detect_trend(self, data, indicators):
        if len(data) < 3:
            return 'Neutral'
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-3]
        if price_change > 0 and data['Close'].iloc[-1] > indicators['ma'].iloc[-1]:
            return 'Bullish'
        if price_change < 0 and data['Close'].iloc[-1] < indicators['ma'].iloc[-1]:
            return 'Bearish'
        return 'Neutral'

    def generate_report(self, data, indicators, signals):
        """Generate formatted trading report"""
        current_time = datetime.datetime.now(self.tz).strftime("%H:%M:%S")
        current_price = data['Close'].iloc[-1]
        session_high = data['High'].max()
        session_low = data['Low'].min()
        
        report = f"""
⚡ **UWS Scalping Alert** ⚡
🕒 {current_time} EST | 📈 ES Futures

💵 **Price Action**
- Current: ${current_price:.2f}
- Session Range: ${session_low:.2f} - ${session_high:.2f}
- VWAP Position: {signals['price_vwap'] or 'N/A'}

📊 **Technical Signals**
- RSI: {indicators['rsi'].iloc[-1]:.1f} ({signals['rsi_status'] or 'N/A'})
- Trend: {signals['trend']}
- Bollinger Squeeze: {'🔔 Active' if signals['bollinger_squeeze'] else '✅ Normal'}
- Volume Spike: {'🚨 Detected' if signals['volume_spike'] else 'Normal'}

🎯 **Key Levels**
- Upper BB: ${indicators['upper_band'].iloc[-1]:.2f}
- Lower BB: ${indicators['lower_band'].iloc[-1]:.2f}
- VWAP: ${indicators['vwap'].iloc[-1]:.2f if indicators['vwap'].any() else 'N/A'}
"""

        return report

    def create_chart(self, data, indicators):
        """Create enhanced technical chart"""
        plt.figure(figsize=(12, 8), facecolor="#1E1E1E")
        ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=4, colspan=1)
        ax2 = plt.subplot2grid((6, 1), (4, 0), rowspan=2, colspan=1, sharex=ax1)

        # Price and Bollinger Bands
        ax1.plot(data.index, data['Close'], label='Price', color='#00FF00', linewidth=1)
        ax1.plot(data.index, indicators['upper_band'], label='Upper BB', color='#FF0000', linestyle='--')
        ax1.plot(data.index, indicators['lower_band'], label='Lower BB', color='#0000FF', linestyle='--')
        ax1.plot(data.index, indicators['vwap'], label='VWAP', color='#FFA500')
        ax1.fill_between(data.index, indicators['lower_band'], indicators['upper_band'], color='#2A2A2A')

        # RSI
        ax2.plot(data.index, indicators['rsi'], label='RSI', color='#00FFFF')
        ax2.axhline(70, color='#FF0000', linestyle='--')
        ax2.axhline(30, color='#00FF00', linestyle='--')

        # Formatting
        ax1.set_title('ES Futures Scalping Dashboard', color='white', fontsize=14)
        ax1.legend(loc='upper left', fontsize=8)
        ax2.set_ylabel('RSI', color='white')
        ax1.set_facecolor('#1E1E1E')
        ax2.set_facecolor('#1E1E1E')
        
        for ax in [ax1, ax2]:
            ax.tick_params(colors='white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.grid(color='#404040')

        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor='#1E1E1E', bbox_inches='tight')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def run_analysis(self):
        """Main analysis workflow"""
        data, data_error = self.fetch_market_data()
        if data_error:
            return {'error': data_error}
            
        indicators, indicator_error = self.calculate_indicators(data)
        if indicator_error:
            return {'error': indicator_error}
            
        signals = self.analyze_price_action(data, indicators)
        report = self.generate_report(data, indicators, signals)
        chart = self.create_chart(data.tail(180), indicators)
        
        return {
            'report': report,
            'chart': chart,
            'data': data,
            'indicators': indicators
        }

def send_discord_alert(webhook_url, message, chart=None):
    """Send formatted alert to Discord"""
    payload = {
        "embeds": [{
            "title": "UWS Scalping Alert",
            "description": message,
            "color": 0x00ff00,
            "image": {"url": "attachment://chart.png"}
        }]
    }
    
    files = {'file': ('chart.png', base64.b64decode(chart), 'image/png')} if chart else None
    try:
        response = requests.post(webhook_url, json=payload if not chart else None, 
                               files=files if chart else None)
        response.raise_for_status()
        logging.info("Alert sent successfully")
    except Exception as e:
        logging.error(f"Discord send error: {str(e)}")

if __name__ == "__main__":
    analyzer = ScalpingAnalyzer()
    results = analyzer.run_analysis()
    
    if 'error' in results:
        send_discord_alert(DISCORD_WEBHOOK_URL, f"❌ Error: {results['error']}")
    else:
        send_discord_alert(DISCORD_WEBHOOK_URL, results['report'], results['chart'])
        logging.info("Analysis completed successfully")
