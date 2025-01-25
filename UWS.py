import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

logging.basicConfig(level=logging.INFO)

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1332276762603683862/aKE2i67QHm-1XR-HsMcQylaS0nKTS4yCVty4-jqvJscwkr6VRTacvLhP89F-4ABFDoQw"
API_KEY = "bbbdc8f307d44bd6bc90f9920926abb4"

# Target run time in Eastern Time (24-hour format)
RUN_HOUR = 11  # 5 PM
RUN_MINUTE = 50

class MarketAnalysis:
    def __init__(self):
        """Initialize analysis with default configurations"""
        self.analysis_config = {
            'period': '1d',  
            'interval': '1m',  
        }
        self.allowed_symbols = ['ES=F']
        self.eastern_tz = pytz.timezone('US/Eastern')
    
    def _generate_advanced_prompt(self, market_data, news_events, market_news):
        """
        Generate a sophisticated, data-driven market analysis prompt
        
        Args:
            market_data (dict): Comprehensive market data
            news_events (list): Upcoming high-impact news events
            market_news (list): Recent market headlines
        
        Returns:
            dict: Structured prompt payload for AI analysis
        """
        try:
            # Prepare technical context
            technical_context = f"""
TECHNICAL SNAPSHOT:
- Current Price: ${market_data['current_price']:.2f}
- Day's Range: ${market_data['session_low']:.2f} - ${market_data['session_high']:.2f}
- Daily Change: {market_data['daily_change']:.2f}%
- Volatility: {market_data['volatility']:.2f}%
- Market Trend: {market_data['market_trend']}
"""
            
            # Prepare news context
            news_context = "MARKET CATALYSTS:\n"
            if news_events:
                for event in sorted(news_events, key=lambda x: x['minutes_until'])[:2]:
                    news_context += f"- {event['impact']} Impact: {event['currency']} {event['event']} in {event['minutes_until']}m\n"
            
            if market_news:
                news_context += "\nRECENT HEADLINES:\n"
                for news in market_news[:2]:
                    news_context += f"- {news['title']}\n"
            
            # Construct comprehensive prompt
            prompt_content = f"""
YOU ARE: A professional quantitative market analyst performing a surgical market analysis.

{technical_context}

{news_context}

ANALYSIS DIRECTIVE:
1. Dissect price action with mathematical precision
2. Identify immediate market structure
3. Assess probabilistic trading scenarios
4. Highlight critical support/resistance levels
5. Determine actionable trading bias

REQUIRED OUTPUT:
- Maximum 250 words
- Use precise numerical references
- Provide probabilistic outcome scenarios
- Include specific entry/exit considerations
- Focus on high-probability market movements
"""
            
            return {
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an elite quantitative market analyst providing high-precision market intelligence."
                    },
                    {
                        "role": "user",
                        "content": prompt_content
                    }
                ],
                "temperature": 0.6,  # Slightly lower temperature for more consistent analysis
                "max_tokens": 300  # Slightly higher to accommodate detailed analysis
            }
        
        except Exception as e:
            logging.error(f"Error generating advanced prompt: {str(e)}")
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
            
            # Existing analysis calculations...
            close_prices = data['Close']
            returns = close_prices.pct_change()
            
            # Previous analysis calculations remain the same...
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

# Rest of the code remains the same...
