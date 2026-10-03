import requests
import re
import datetime
from .config import get_api_key, get_config


def get_news(symbol: str, n: int = 5):
    """
    Fetches news for a given cryptocurrency symbol from CryptoCompare API.

    Args:
        symbol (str): The cryptocurrency symbol (e.g., 'BTC', 'ETH').
        n (int): The number of sentences to extract from the body of each news article.

    Returns:
        str: A formatted string of news articles, or an error message string if something goes wrong.
    """
    api_key = get_api_key("coindesk_api_key", "COINDESK_API_KEY")
    if not api_key:
        print("COINDESK_API_KEY not found in environment variables.")
        return "COINDESK_API_KEY not found in environment variables."

    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories={symbol}"

    headers = {"Authorization": f"Apikey {api_key}"}

    try:
        response = requests.get(url, headers=headers, timeout=get_config().get("news_request_timeout_seconds", 20))
        response.raise_for_status()
        news_data = response.json()

        if news_data.get("Type") != 100 or not news_data.get("Data"):
            return f"No news found for {symbol}."

        formatted_news = []
        for article in news_data["Data"]:
            title = article.get("title", "No Title")
            source = article.get("source_info", {}).get("name", "Unknown Source")
            body = article.get("body", "")
            
            # Get and format the published timestamp
            published_timestamp = article.get("published_on", 0)
            published_date = datetime.datetime.fromtimestamp(published_timestamp).strftime('%Y-%m-%d %H:%M:%S')

            # Split body into sentences and take the first N.
            sentences = re.split(r"(?<=[.!?])\s+", body)
            summary = " ".join(sentences[:n])

            formatted_news.append(f"### {title} (source: {source}, published: {published_date})\n\n{summary}\n\n")

        return "".join(formatted_news)

    except requests.exceptions.RequestException as e:
        return f"Error fetching news from CryptoCompare: {e}"
    except Exception as e:
        return f"An error occurred: {e}"
