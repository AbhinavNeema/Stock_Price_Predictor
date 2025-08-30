import os
import time
import numpy as np
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
analyzer = SentimentIntensityAnalyzer()
load_dotenv()
NEWS_API_KEY = "f4720ca914da4e1eba03a3f520aa17f7"

def _safe_float(x, default=0.0):
    """Convert x to a plain Python float safe for JSON; replace NaN/inf/None with default."""
    try:
        f = float(x)
        if np.isnan(f) or np.isinf(f):
            return float(default)
        return float(f)
    except Exception:
        return float(default)

def get_live_sentiment(stock_ticker):
    """Fetches recent news for a stock and calculates a single sentiment score."""
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWS_API_KEY":
        return 0.0
    query = stock_ticker.split('.')[0]
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize=20"
    try:
        response = requests.get(url, timeout=8)
        articles = response.json().get('articles', [])
        if not articles:
            return 0.0
        content_to_analyze = [f"{a.get('title','')}. {a.get('description','')}" for a in articles if a.get('title') and a.get('description')]
        if not content_to_analyze:
            return 0.0
        compound_scores = [analyzer.polarity_scores(content)['compound'] for content in content_to_analyze]
        return float(np.mean(compound_scores)) if compound_scores else 0.0
    except Exception as e:
        print(f"  - Could not fetch news for {stock_ticker}: {e}")
        return 0.0
