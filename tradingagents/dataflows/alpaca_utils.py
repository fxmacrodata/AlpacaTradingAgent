# alpaca_utils.py

import math
import os
import pandas as pd
import time
import threading
from datetime import datetime, timedelta
from typing import Annotated, Union, Optional, List, Dict, Any, TYPE_CHECKING
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest, StockLatestQuoteRequest, CryptoLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetAssetsRequest,
    GetOrdersRequest,
    MarketOrderRequest,
    ClosePositionRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from alpaca.trading.enums import AssetClass, AssetStatus, OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.common.enums import Sort
from .config import get_api_key, get_alpaca_use_paper, get_config
from .ticker_utils import TickerUtils
# Imported lazily inside execute_trade_intent: a module-level import would
# create a circular import (dataflows -> agents -> dataflows.interface).
if TYPE_CHECKING:
    from tradingagents.agents.schemas import TradeIntent

from tradingagents.risk.position_sizing import (
    PositionSizer,
    RiskParameters,
    SizingDecision,
    compute_atr,
)

# Serialize entry reconciliation and submission within this process. Broker
# open orders also protect subsequent loop iterations and process restarts.
_ENTRY_ORDER_LOCK = threading.Lock()


class PendingEntryError(RuntimeError):
    pass


def _submit_entry_order(client, request):
    with _ENTRY_ORDER_LOCK:
        pending = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN, symbols=[request.symbol], limit=1,
        ))
        if pending:
            raise PendingEntryError(f"Open order already exists for {request.symbol}; entry deferred.")
        return client.submit_order(request)


# Fallback dictionary for company names
ticker_to_company_fallback = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "TSM": "Taiwan Semiconductor Manufacturing Company OR TSMC",
    "JPM": "JPMorgan Chase OR JP Morgan",
    "JNJ": "Johnson & Johnson OR JNJ",
    "V": "Visa",
    "WMT": "Walmart",
    "META": "Meta OR Facebook",
    "AMD": "AMD",
    "INTC": "Intel",
    "QCOM": "Qualcomm",
    "BABA": "Alibaba",
    "ADBE": "Adobe",
    "NFLX": "Netflix",
    "CRM": "Salesforce",
    "PYPL": "PayPal",
    "VZ": "Verizon OR Verizon Communications",
    "PLTR": "Palantir",
    "MU": "Micron",
    "SQ": "Block OR Square",
    "ZM": "Zoom",
    "CSCO": "Cisco",
    "SHOP": "Shopify",
    "ORCL": "Oracle",
    "X": "Twitter OR X",
    "SPOT": "Spotify",
    "AVGO": "Broadcom",
    "ASML": "ASML ",
    "TWLO": "Twilio",
    "SNAP": "Snap Inc.",
    "TEAM": "Atlassian",
    "SQSP": "Squarespace",
    "UBER": "Uber",
    "ROKU": "Roku",
    "PINS": "Pinterest",
}


_ASSET_SEARCH_CACHE = {
    "expires_at": 0.0,
    "assets": [],
}


def _enum_value(value) -> str:
    return getattr(value, "value", value) or ""


def _normalize_crypto_symbol(symbol: str) -> str:
    raw = (symbol or "").upper().replace("-", "/")
    if "/" in raw:
        return raw
    for quote in ("USDT", "USDC", "USD"):
        if raw.endswith(quote) and len(raw) > len(quote):
            return f"{raw[:-len(quote)]}/{quote}"
    return raw


def _normalize_asset_symbol(symbol: str, asset_class: str) -> str:
    raw = (symbol or "").upper()
    if asset_class == AssetClass.CRYPTO.value:
        return _normalize_crypto_symbol(raw)
    return raw


def _asset_to_search_result(asset) -> Dict[str, Any]:
    asset_class = _enum_value(getattr(asset, "asset_class", "")) or "unknown"
    symbol = _normalize_asset_symbol(getattr(asset, "symbol", ""), asset_class)
    name = getattr(asset, "name", "") or ticker_to_company_fallback.get(symbol, symbol)
    exchange = _enum_value(getattr(asset, "exchange", "")) or ""
    tradable = bool(getattr(asset, "tradable", False))
    asset_type = "Crypto" if asset_class == AssetClass.CRYPTO.value else "Equity"
    return {
        "symbol": symbol,
        "name": name,
        "asset_class": asset_class,
        "asset_type": asset_type,
        "exchange": exchange,
        "tradable": tradable,
        "market_cap": None,
    }


def _fallback_asset_results() -> List[Dict[str, Any]]:
    stocks = [
        ("NVDA", "NVIDIA Corporation", "NASDAQ"),
        ("AMD", "Advanced Micro Devices, Inc.", "NASDAQ"),
        ("TSLA", "Tesla, Inc.", "NASDAQ"),
        ("AAPL", "Apple Inc.", "NASDAQ"),
        ("MSFT", "Microsoft Corporation", "NASDAQ"),
    ]
    crypto = [
        ("BTC/USD", "Bitcoin / US Dollar"),
        ("ETH/USD", "Ethereum / US Dollar"),
        ("SOL/USD", "Solana / US Dollar"),
    ]
    results = [
        {
            "symbol": symbol,
            "name": name,
            "asset_class": AssetClass.US_EQUITY.value,
            "asset_type": "Equity",
            "exchange": exchange,
            "tradable": True,
            "market_cap": None,
        }
        for symbol, name, exchange in stocks
    ]
    results.extend(
        {
            "symbol": symbol,
            "name": name,
            "asset_class": AssetClass.CRYPTO.value,
            "asset_type": "Crypto",
            "exchange": "Alpaca Crypto",
            "tradable": True,
            "market_cap": None,
        }
        for symbol, name in crypto
    )
    return results


def get_alpaca_stock_client() -> StockHistoricalDataClient:
    api_key = get_api_key("alpaca_api_key", "ALPACA_API_KEY")
    api_secret = get_api_key("alpaca_secret_key", "ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        print(f"Warning: Missing Alpaca API credentials. API key: {'present' if api_key else 'missing'}, Secret: {'present' if api_secret else 'missing'}")
        raise ValueError("Alpaca API key or secret not found. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
    try:
        return StockHistoricalDataClient(api_key, api_secret)
    except Exception as e:
        print(f"Error creating Alpaca stock client: {e}")
        raise


def get_alpaca_crypto_client() -> CryptoHistoricalDataClient:
    api_key = get_api_key("alpaca_api_key", "ALPACA_API_KEY")
    api_secret = get_api_key("alpaca_secret_key", "ALPACA_SECRET_KEY")
    # Crypto calls work without keys, but keys raise rate limits
    if api_key and api_secret:
        return CryptoHistoricalDataClient(api_key, api_secret)
    else:
        return CryptoHistoricalDataClient()


def get_alpaca_trading_client() -> TradingClient:
    api_key = get_api_key("alpaca_api_key", "ALPACA_API_KEY")
    api_secret = get_api_key("alpaca_secret_key", "ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise ValueError("Alpaca API key or secret not found. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
    use_paper = str(get_alpaca_use_paper() or "True").strip().lower() not in ("false", "0", "no")
    return TradingClient(api_key, api_secret, paper=use_paper)


def _parse_timeframe(tf: Union[str, TimeFrame]) -> TimeFrame:
    """Convert a string like '5Min' or a TimeFrame instance into a TimeFrame."""
    if isinstance(tf, TimeFrame):
        return tf

    tf = tf.strip()
    low = tf.lower()
    
    # mapping common strings
    if low == "1min":
        result = TimeFrame.Minute
    elif low.endswith("min"):
        # e.g. "5Min", "15min"
        amount = int(tf[:-3])
        result = TimeFrame(amount, TimeFrameUnit.Minute)
    elif low == "1hour" or low == "1h":
        result = TimeFrame.Hour
    elif low.endswith("hour"):
        amount = int(tf[:-4])
        result = TimeFrame(amount, TimeFrameUnit.Hour)
    elif low.endswith("h") and low[:-1].isdigit():
        # shorthand: "4h", "2h", etc.
        amount = int(low[:-1])
        result = TimeFrame(amount, TimeFrameUnit.Hour)
    elif low == "1day" or low == "1d":
        result = TimeFrame.Day
    elif low.endswith("day"):
        amount = int(tf[:-3])
        result = TimeFrame(amount, TimeFrameUnit.Day)
    elif low.endswith("d") and low[:-1].isdigit():
        # shorthand: "2d", "3d", etc.
        amount = int(low[:-1])
        result = TimeFrame(amount, TimeFrameUnit.Day)
    else:
        # fallback
        result = TimeFrame.Day
    
    return result


def _is_supported_data_fallback_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "subscription",
            "permission",
            "unauthorized",
            # nginx-level 401s arrive as HTML pages with this phrasing
            "authorization required",
            "forbidden",
            "not found",
            "empty",
            "rate limit",
            "too many requests",
            "timeout",
            "api key",
            "secret not found",
        )
    )


def _yfinance_fallback_data(
    symbol: str,
    start: pd.Timestamp,
    end: Optional[pd.Timestamp],
    timeframe: Union[str, TimeFrame],
) -> pd.DataFrame:
    config = get_config()
    if not config.get("data_fallback_enabled", False):
        return pd.DataFrame()

    tf_text = str(timeframe).lower()
    interval = "1d"
    if "hour" in tf_text or tf_text in ("1h",):
        interval = "1h"
    elif "min" in tf_text:
        return pd.DataFrame()

    try:
        import yfinance as yf

        yahoo_symbol = TickerUtils.convert_for_api(symbol, "yahoo")
        data = yf.download(
            yahoo_symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d") if end is not None else None,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as exc:
        print(f"YFinance fallback failed for {symbol}: {exc}")
        return pd.DataFrame()

    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    rename_map = {
        "Date": "timestamp",
        "Datetime": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = data.reset_index().rename(columns=rename_map)
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"YFinance fallback returned malformed data for {symbol}: missing {missing}")
        return pd.DataFrame()
    return df[required].dropna().reset_index(drop=True)


class AlpacaUtils:
    @staticmethod
    def _get_searchable_assets(cache_seconds: int = 900) -> List[Dict[str, Any]]:
        """Return active equity and crypto assets, cached to keep symbol search responsive."""
        now = time.time()
        cached_assets = _ASSET_SEARCH_CACHE.get("assets") or []
        if cached_assets and now < _ASSET_SEARCH_CACHE.get("expires_at", 0):
            return cached_assets

        try:
            client = get_alpaca_trading_client()
            assets = []
            for asset_class in (AssetClass.US_EQUITY, AssetClass.CRYPTO):
                request = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=asset_class)
                assets.extend(client.get_all_assets(request))

            searchable_assets = []
            seen_symbols = set()
            for asset in assets:
                result = _asset_to_search_result(asset)
                symbol = result["symbol"]
                if not symbol or symbol in seen_symbols:
                    continue
                seen_symbols.add(symbol)
                searchable_assets.append(result)

            _ASSET_SEARCH_CACHE["assets"] = searchable_assets
            _ASSET_SEARCH_CACHE["expires_at"] = now + cache_seconds
            return searchable_assets
        except Exception as e:
            print(f"Error loading Alpaca assets for search: {e}")
            fallback = _fallback_asset_results()
            _ASSET_SEARCH_CACHE["assets"] = fallback
            _ASSET_SEARCH_CACHE["expires_at"] = now + 60
            return fallback

    @staticmethod
    def search_assets(query: str = "", limit: int = 12) -> List[Dict[str, Any]]:
        """Search active Alpaca equity and crypto assets for the WebUI symbol picker."""
        query = (query or "").strip().upper()
        normalized_query = query.replace("/", "").replace("-", "")
        assets = AlpacaUtils._get_searchable_assets()

        if not normalized_query:
            default_symbols = ["NVDA", "AMD", "TSLA", "AAPL", "MSFT", "BTC/USD", "ETH/USD", "SOL/USD"]
            by_symbol = {asset["symbol"]: asset for asset in assets}
            return [by_symbol.get(symbol) or asset for symbol in default_symbols for asset in _fallback_asset_results() if asset["symbol"] == symbol][:limit]

        matches = []
        for asset in assets:
            symbol = asset["symbol"]
            symbol_key = symbol.replace("/", "")
            crypto_base = symbol.split("/", 1)[0] if asset.get("asset_class") == AssetClass.CRYPTO.value else ""
            crypto_quote = symbol.split("/", 1)[1] if "/" in symbol else ""
            name = (asset.get("name") or "").upper()
            if crypto_base and normalized_query == crypto_base:
                quote_priority = {"USD": 0, "USDC": 0.1, "USDT": 0.2}.get(crypto_quote, 0.4)
                score = -1 + quote_priority
            elif normalized_query == symbol_key:
                score = 0
            elif symbol_key.startswith(normalized_query):
                score = 1
            elif name.startswith(query):
                score = 2
            elif normalized_query in symbol_key or query in name:
                score = 3
            else:
                continue

            tradable_penalty = 0 if asset.get("tradable") else 1
            crypto_bonus = 0 if asset.get("asset_class") != AssetClass.CRYPTO.value else -0.1
            matches.append((score + tradable_penalty + crypto_bonus, symbol, asset))

        matches.sort(key=lambda item: (item[0], item[1]))
        return [asset for _, _, asset in matches[:limit]]

    @staticmethod
    def get_stock_data(
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None,
        timeframe: Union[str, TimeFrame] = "1Day",
        save_path: Optional[str] = None,
        feed: DataFeed = DataFeed.IEX
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a stock or crypto symbol.

        Args:
            symbol: The ticker symbol (e.g. "SPY" or "BTC/USD")
            start_date: 'YYYY-MM-DD' string or datetime
            end_date: optional 'YYYY-MM-DD' string or datetime
            timeframe: e.g. "1Min","5Min","15Min","1Hour","1Day" or a TimeFrame instance
            save_path: if provided, path to write a CSV
            feed: DataFeed enum (default IEX)

        Returns:
            pandas DataFrame with columns ['timestamp','open','high','low','close','volume']
        """
        # normalize dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) + timedelta(days=1) if end_date else None

        tf = _parse_timeframe(timeframe)

        try:
            # choose client
            is_crypto = "/" in symbol
            client = get_alpaca_crypto_client() if is_crypto else get_alpaca_stock_client()

            # build request params; always use a list for symbol_or_symbols
            params = (
                CryptoBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=tf,
                    start=start,
                    end=end,
                    feed=feed
                ) if is_crypto else
                StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=tf,
                    start=start,
                    end=end,
                    feed=feed
                )
            )
            bars = client.get_crypto_bars(params) if is_crypto else client.get_stock_bars(params)
            # convert to DataFrame via the .df property
            df = bars.df.reset_index()  # multi-index ['symbol','timestamp']
            
            # filter for our symbol (in case of list) - only if symbol column exists
            if "symbol" in df.columns:
                df = df[df["symbol"] == symbol].drop(columns="symbol")
            else:
                # If no symbol column, assume all data is for the requested symbol
                pass

            # Alpaca can include a bar at the exact end boundary. The public
            # date range includes end_date, but must exclude the next day's
            # midnight bar (especially for daily crypto data).
            if end is not None and "timestamp" in df.columns:
                cutoff = pd.Timestamp(end)
                cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
                df = df.loc[pd.to_datetime(df["timestamp"], utc=True) < cutoff].reset_index(drop=True)
                
            if df.empty:
                raise ValueError("empty Alpaca bar response")

            if save_path:
                df.to_csv(save_path, index=False, encoding="utf-8")
            return df

        except Exception as e:
            if _is_supported_data_fallback_error(e):
                fallback_df = _yfinance_fallback_data(symbol, start, end, timeframe)
                if not fallback_df.empty:
                    if save_path:
                        fallback_df.to_csv(save_path, index=False, encoding="utf-8")
                    return fallback_df
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_latest_quote(symbol: str) -> dict:
        """
        Get the latest bid/ask quote for a symbol.
        """
        is_crypto = "/" in symbol
        client = get_alpaca_crypto_client() if is_crypto else get_alpaca_stock_client()
        req = CryptoLatestQuoteRequest(symbol_or_symbols=[symbol]) if is_crypto else StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        try:
            resp = client.get_crypto_latest_quote(req) if is_crypto else client.get_stock_latest_quote(req)
            quote = resp[symbol]
            return {
                "symbol": symbol,
                "bid_price": quote.bid_price,
                "bid_size": quote.bid_size,
                "ask_price": quote.ask_price,
                "ask_size": quote.ask_size,
                "timestamp": quote.timestamp
            }
        except Exception as e:
            print(f"Error fetching latest quote for {symbol}: {e}")
            return {}

    
    @staticmethod
    def get_stock_data_window(
        symbol: Annotated[str, "ticker symbol"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"] = None,
        look_back_days: Annotated[int, "Number of days to look back"] = 30,
        timeframe: Annotated[str, "Timeframe for data: 1Min, 5Min, 15Min, 1Hour, 1Day"] = "1Day",
    ) -> pd.DataFrame:
        """
        Fetches historical stock data from Alpaca for the specified symbol and a window of days.
        
        Args:
            symbol: The stock ticker symbol
            curr_date: Current date in yyyy-mm-dd format (optional - if not provided, will use today's date)
            look_back_days: Number of days to look back
            timeframe: Timeframe for data (1Min, 5Min, 15Min, 1Hour, 1Day)
            
        Returns:
            DataFrame containing the historical stock data
        """
        # Calculate start date based on look_back_days
        if curr_date:
            curr_dt = pd.to_datetime(curr_date)
        else:
            curr_dt = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
            
        start_dt = curr_dt - pd.Timedelta(days=look_back_days)
        
        # Don't pass end_date to avoid subscription limitations
        return AlpacaUtils.get_stock_data(
            symbol=symbol,
            start_date=start_dt.strftime("%Y-%m-%d"),
            timeframe=timeframe
        ) 

    @staticmethod
    def get_company_name(symbol: str) -> str:
        """
        Get company name for a ticker symbol using Alpaca API.
        
        Args:
            symbol: The ticker symbol (e.g. "AAPL")
            
        Returns:
            Company name as string or original symbol if not found
        """
        try:
            # Skip crypto or symbols with special characters
            if "/" in symbol:
                return symbol
                
            client = get_alpaca_trading_client()
            asset = client.get_asset(symbol)
            
            if asset and hasattr(asset, 'name') and asset.name:
                return asset.name
            else:
                # Use fallback if name is not available
                print(f"No company name found for {symbol} via API, using fallback.")
                return ticker_to_company_fallback.get(symbol, symbol)
                
        except Exception as e:
            print(f"Error fetching company name for {symbol}: {e}")
            print("This might be due to invalid API keys or insufficient permissions.")
            print("If you recently reset your paper trading account, you may need to generate new API keys.")
            return ticker_to_company_fallback.get(symbol, symbol) 

    @staticmethod
    def get_positions_data():
        """Get current positions from Alpaca account"""
        try:
            client = get_alpaca_trading_client()
            positions = client.get_all_positions()
            
            # Convert positions to a list of dictionaries
            positions_data = []
            for position in positions:
                current_price = float(position.current_price)
                avg_entry_price = float(position.avg_entry_price)
                qty = float(position.qty)
                market_value = float(position.market_value)
                cost_basis = avg_entry_price * qty
                
                # Calculate P/L values
                today_pl_dollars = float(position.unrealized_intraday_pl)
                total_pl_dollars = float(position.unrealized_pl)
                today_pl_percent = (today_pl_dollars / cost_basis) * 100 if cost_basis != 0 else 0
                total_pl_percent = (total_pl_dollars / cost_basis) * 100 if cost_basis != 0 else 0
                
                positions_data.append({
                    "Symbol": position.symbol,
                    "Qty": qty,
                    "Market Value": f"${market_value:.2f}",
                    "Avg Entry": f"${avg_entry_price:.2f}",
                    "Cost Basis": f"${cost_basis:.2f}",
                    "Today's P/L (%)": f"{today_pl_percent:.2f}%",
                    "Today's P/L ($)": f"${today_pl_dollars:.2f}",
                    "Total P/L (%)": f"{total_pl_percent:.2f}%",
                    "Total P/L ($)": f"${total_pl_dollars:.2f}"
                })
            
            return positions_data
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []

    @staticmethod
    def get_recent_orders(page=1, page_size=7):
        """Get recent orders from Alpaca account, with simple pagination."""
        return AlpacaUtils.get_recent_orders_page(page=page, page_size=page_size).get("orders", [])

    @staticmethod
    def get_recent_orders_page(page=1, page_size=5, max_orders=500):
        """Get recent Alpaca orders and pagination metadata for the WebUI."""
        try:
            client = get_alpaca_trading_client()
            req = GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                limit=max_orders,
                direction=Sort.DESC,
                nested=False,
            )
            orders_page = client.get_orders(req)
            orders = list(orders_page)

            orders_data = []
            for order in orders:
                qty = float(order.qty) if order.qty is not None else 0.0
                filled_qty = float(order.filled_qty) if order.filled_qty is not None else 0.0
                filled_avg_price = float(order.filled_avg_price) if order.filled_avg_price is not None else 0.0

                orders_data.append({
                    "Asset": order.symbol,
                    "Order Type": order.type,
                    "Side": order.side,
                    "Qty": qty,
                    "Filled Qty": filled_qty,
                    "Avg. Fill Price": f"${filled_avg_price:.2f}" if filled_avg_price > 0 else "-",
                    "Status": order.status,
                    "Source": order.client_order_id
                })

            total_orders = len(orders_data)
            total_pages = max(1, (total_orders + page_size - 1) // page_size)
            page = max(1, min(int(page or 1), total_pages))
            start = (page - 1) * page_size
            return {
                "orders": orders_data[start : start + page_size],
                "page": page,
                "page_size": page_size,
                "total_orders": total_orders,
                "total_pages": total_pages,
                "has_more": total_orders >= max_orders,
            }

        except Exception as e:
            print(f"Error fetching orders: {e}")
            return {
                "orders": [],
                "page": max(1, int(page or 1)),
                "page_size": page_size,
                "total_orders": 0,
                "total_pages": 1,
                "has_more": False,
            }

    @staticmethod
    def get_account_info():
        """Get account information from Alpaca"""
        try:
            client = get_alpaca_trading_client()
            account = client.get_account()
            
            # Extract the required values
            buying_power = float(account.buying_power)
            cash = float(account.cash)
            
            # Calculate daily change
            equity = float(account.equity)
            last_equity = float(account.last_equity)
            daily_change_dollars = equity - last_equity
            daily_change_percent = (daily_change_dollars / last_equity) * 100 if last_equity != 0 else 0
            
            return {
                "buying_power": buying_power,
                "cash": cash,
                "daily_change_dollars": daily_change_dollars,
                "daily_change_percent": daily_change_percent
            }
        except Exception as e:
            print(f"Error fetching account info: {e}")
            return {
                "buying_power": 0,
                "cash": 0,
                "daily_change_dollars": 0,
                "daily_change_percent": 0
            } 

    @staticmethod
    def get_current_position_state(symbol: str, strict: bool = False) -> str:
        """Return current position state for a symbol in the Alpaca account.

        Args:
            symbol: Ticker symbol (e.g. "AAPL" or "BTC/USD").  Crypto symbols will
                    be treated the same way as equities – a positive quantity is
                    considered a *LONG* position while a negative quantity (should
                    Alpaca ever allow it) is considered *SHORT*.
            strict: When True, re-raise broker/API errors instead of defaulting
                    to "NEUTRAL".  The NEUTRAL fallback exists so agent prompts
                    keep working through an outage; order execution paths must
                    pass strict=True, because acting on a guessed NEUTRAL can
                    re-buy an existing position or silently skip an exit.

        Returns:
            One of "LONG", "SHORT", or "NEUTRAL" if no open position exists (or,
            when strict is False, when we encounter an error).
        """
        try:
            # Skip if credentials are missing – the helper will raise inside but we
            # want to fail gracefully and just assume no position.
            client = get_alpaca_trading_client()

            # `get_all_positions()` is more broadly supported across Alpaca
            # versions than `get_position(symbol)` and avoids raising when the
            # asset is not found.
            positions = client.get_all_positions()

            # Normalise the requested symbol for comparisons – Alpaca symbols
            # for crypto may use different formats, so we normalize for position comparison only.
            requested_symbol_key = symbol.upper().replace("/", "")

            for pos in positions:
                if pos.symbol.upper() == requested_symbol_key:
                    try:
                        qty = float(pos.qty)
                    except (ValueError, AttributeError):
                        if strict:
                            # A corrupted qty on the *target* position is as
                            # unsafe as an outage for an execution caller:
                            # guessing NEUTRAL here re-buys a real holding or
                            # skips a real exit. Let it propagate.
                            raise
                        qty = 0.0

                    if not math.isfinite(qty):
                        # e.g. qty "nan"/"inf" parses without raising but is
                        # not a real position size. Same fail-open risk as a
                        # malformed string for an execution caller.
                        if strict:
                            raise ValueError(
                                f"non-finite position qty for {symbol}: {pos.qty!r}"
                            )
                        return "NEUTRAL"

                    if qty > 0:
                        return "LONG"
                    elif qty < 0:
                        return "SHORT"
                    else:
                        # Zero quantity technically shouldn't appear but treat as
                        # neutral just in case.
                        return "NEUTRAL"
            # If we fall through the loop there is no open position for symbol.
            return "NEUTRAL"
        except Exception as e:
            if strict:
                # Execution callers must not mistake an outage for "no
                # position": a wrong NEUTRAL turns a BUY into pyramiding an
                # existing holding and a SELL into a skipped exit.
                raise
            # Log and default to neutral so agent prompts still work.
            print(f"Error determining current position for {symbol}: {e}")
            return "NEUTRAL"

    @staticmethod
    def place_market_order(symbol: str, side: str, notional: float = None, qty: float = None) -> dict:
        """
        Place a market order with Alpaca
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            side: "buy" or "sell"
            notional: Dollar amount to buy/sell (for fractional shares)
            qty: Number of shares (if not using notional)
            
        Returns:
            Dictionary with order result information
        """
        try:
            client = get_alpaca_trading_client()
            
            # Normalize symbol for Alpaca (remove "/" for crypto)
            alpaca_symbol = symbol.upper().replace("/", "")
            
            # Determine order side
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            
            # Determine proper time-in-force: crypto orders only allow GTC
            is_crypto = "/" in symbol.upper()
            tif = TimeInForce.GTC if is_crypto else TimeInForce.DAY

            # Create market order request
            if notional and notional > 0:
                # Use notional (dollar amount) for fractional shares
                order_request = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    side=order_side,
                    time_in_force=tif,
                    notional=notional
                )
            elif qty and qty > 0:
                # Use quantity (number of shares)
                order_request = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    side=order_side,
                    time_in_force=tif,
                    qty=qty
                )
            else:
                return {"success": False, "error": "Must specify either notional or qty"}
            
            # Submit the order
            order = _submit_entry_order(client, order_request)
            
            return {
                "success": True,
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": float(order.qty) if order.qty else None,
                "notional": float(order.notional) if order.notional else None,
                "status": order.status,
                "message": f"Successfully placed {side} order for {symbol}"
            }
            
        except PendingEntryError as e:
            return {"success": False, "broker_attempted": False, "error": str(e)}
        except Exception as e:
            error_msg = f"Error placing {side} order for {symbol}: {e}"
            print(error_msg)
            return {"success": False, "error": error_msg}

    @staticmethod
    def place_protected_market_order(
        symbol: str,
        side: str,
        qty: float,
        stop_loss_price: float = None,
        take_profit_price: float = None,
    ) -> dict:
        """Place a market order with broker-side protective child orders.

        Uses order_class=bracket when both a stop and a target are given, and
        order_class=oto for a single protective leg.  Equities only: Alpaca
        does not support bracket/OTO orders for crypto, and protective legs
        require a whole-share quantity (no notional sizing).  Time in force is
        GTC so the protective legs survive past the trading day.
        """
        try:
            if not stop_loss_price and not take_profit_price:
                return {"success": False, "error": "No protective price supplied"}

            client = get_alpaca_trading_client()
            alpaca_symbol = symbol.upper().replace("/", "")
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            stop_loss = (
                StopLossRequest(stop_price=round(float(stop_loss_price), 2))
                if stop_loss_price
                else None
            )
            take_profit = (
                TakeProfitRequest(limit_price=round(float(take_profit_price), 2))
                if take_profit_price
                else None
            )
            order_class = (
                OrderClass.BRACKET if (stop_loss and take_profit) else OrderClass.OTO
            )

            order_request = MarketOrderRequest(
                symbol=alpaca_symbol,
                side=order_side,
                time_in_force=TimeInForce.GTC,
                qty=int(qty),
                order_class=order_class,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            order = _submit_entry_order(client, order_request)

            return {
                "success": True,
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": float(order.qty) if order.qty else None,
                "status": order.status,
                "order_class": "bracket" if order_class == OrderClass.BRACKET else "oto",
                "stop_loss_price": float(stop_loss.stop_price) if stop_loss else None,
                "take_profit_price": float(take_profit.limit_price) if take_profit else None,
                "message": (
                    f"Placed {side} {order_class.value} order for {symbol} "
                    f"(stop={stop_loss.stop_price if stop_loss else None}, "
                    f"target={take_profit.limit_price if take_profit else None})"
                ),
            }

        except PendingEntryError as e:
            return {"success": False, "broker_attempted": False, "error": str(e)}
        except Exception as e:
            error_msg = f"Error placing protected {side} order for {symbol}: {e}"
            print(error_msg)
            return {"success": False, "error": error_msg}

    @staticmethod
    def close_position(symbol: str, percentage: float = 100.0) -> dict:
        """
        Close a position (partially or completely)
        
        Args:
            symbol: Stock symbol
            percentage: Percentage of position to close (default 100% = full close)
            
        Returns:
            Dictionary with close result information
        """
        try:
            percentage = float(percentage)
            if not math.isfinite(percentage) or not 0 < percentage <= 100:
                return {"success": False, "broker_attempted": False,
                        "error": "Close percentage must be greater than 0 and at most 100."}
            client = get_alpaca_trading_client()
            
            # Normalize symbol for Alpaca
            alpaca_symbol = symbol.upper().replace("/", "")
            
            # For full position close (100%), don't specify percentage - let Alpaca close entire position
            if percentage >= 100.0:
                # Close the entire position without specifying percentage
                order = client.close_position(alpaca_symbol)
            else:
                # Create close position request for partial close
                close_request = ClosePositionRequest(
                    percentage=str(percentage)
                )
                order = client.close_position(alpaca_symbol, close_request)
            
            return {
                "success": True,
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": float(order.qty) if order.qty else None,
                "status": order.status,
                "message": f"Submitted order to close {percentage}% of {symbol} position"
            }
            
        except Exception as e:
            error_msg = f"Error closing position for {symbol}: {e}"
            print(error_msg)
            return {"success": False, "error": error_msg}

    @staticmethod
    def get_account_risk_snapshot() -> dict:
        """Numeric account snapshot for the deterministic risk engine.

        Unlike get_account_info, this raises on API failure instead of
        returning zeros: silent zero equity would read as "no capital" and
        corrupt every downstream sizing decision.
        """
        client = get_alpaca_trading_client()
        account = client.get_account()
        equity = float(account.equity)
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("Account equity is unavailable or invalid")
        gross_exposure = 0.0
        for position in client.get_all_positions():
            value = float(position.market_value)
            if not math.isfinite(value):
                raise ValueError("Position exposure is unavailable or invalid")
            gross_exposure += abs(value)
        return {"equity": equity, "gross_exposure": gross_exposure}

    @staticmethod
    def compute_risk_sized_amount(
        symbol: str,
        confidence: str,
        requested_notional: float,
        risk_params: Optional[dict] = None,
        side: str = "buy",
    ) -> SizingDecision:
        """Run the deterministic sizing engine against live account/market data.

        Raises when required data (account snapshot, price) is unavailable so
        the caller can decide whether to fail open or block.
        """
        params = RiskParameters.from_dict(risk_params)
        snapshot = AlpacaUtils.get_account_risk_snapshot()

        quote = AlpacaUtils.get_latest_quote(symbol)
        quoted = [
            float(p)
            for p in (quote.get("bid_price"), quote.get("ask_price"))
            if p and float(p) > 0
        ]
        price = sum(quoted) / len(quoted) if quoted else None

        bars = AlpacaUtils.get_stock_data_window(
            symbol, look_back_days=max(40, params.atr_period * 3)
        )
        atr = compute_atr(bars, period=params.atr_period)

        if price is None:
            try:
                price = float(bars["close"].iloc[-1])
            except Exception:
                raise ValueError(f"No price data available for {symbol}")

        return PositionSizer(params).size_position(
            equity=snapshot["equity"],
            price=price,
            atr=atr,
            confidence=confidence,
            requested_notional=requested_notional,
            current_gross_exposure=snapshot["gross_exposure"],
            side=side,
        )

    @staticmethod
    def execute_trade_intent(
        symbol: str,
        current_position: str,
        trade_intent: Union["TradeIntent", Dict[str, Any]],
        dollar_amount: float,
        allow_shorts: bool = False,
        risk_params: Optional[dict] = None,
    ) -> dict:
        """Validate and execute a typed TradeIntent.

        Numeric equity stops/targets are submitted as bracket/OTO orders.
        A failed sizing or protected submission must not enlarge or strip
        protection from the requested exposure.
        """
        # Imported lazily: a module-level import of the agents package from
        # here creates a dataflows <-> agents import cycle that breaks
        # whenever dataflows is imported first.
        from tradingagents.agents.schemas import TradeIntent, trade_intent_action

        try:
            intent = (
                trade_intent
                if isinstance(trade_intent, TradeIntent)
                else TradeIntent.model_validate(trade_intent)
            )
        except Exception as e:
            return {"success": False, "error": f"Invalid trade intent: {e}"}

        requested_symbol = (intent.symbol or "").upper().replace("/", "")
        actual_symbol = (symbol or "").upper().replace("/", "")
        if requested_symbol and requested_symbol != actual_symbol:
            return {
                "success": False,
                "error": f"Trade intent symbol {intent.symbol} does not match execution symbol {symbol}",
                "trade_intent": intent.model_dump(mode="json"),
            }

        signal = trade_intent_action(intent)
        if not signal:
            return {
                "success": False,
                "error": "Trade intent does not contain an executable action",
                "trade_intent": intent.model_dump(mode="json"),
            }

        mode = (intent.trading_mode or "investment").lower()
        allowed_actions = (
            {"LONG", "NEUTRAL", "SHORT"}
            if mode == "trading"
            else {"BUY", "HOLD", "SELL"}
        )
        if signal not in allowed_actions:
            return {
                "success": False,
                "error": f"Action {signal} is invalid for {mode} mode",
                "trade_intent": intent.model_dump(mode="json"),
            }

        is_crypto = "/" in symbol.upper()
        if signal == "SHORT" and (is_crypto or not allow_shorts):
            reason = (
                "Crypto short exposure is not supported by Alpaca spot trading"
                if is_crypto
                else "Short exposure is disabled for this session"
            )
            return {
                "success": False,
                "error": reason,
                "trade_intent": intent.model_dump(mode="json"),
            }

        warnings = list(intent.execution_constraints.warnings)
        if intent.current_position.value != current_position:
            warnings.append(
                f"Intent was generated with {intent.current_position.value} position; live position is {current_position}."
            )

        target_position = {
            "BUY": "LONG",
            "LONG": "LONG",
            "SHORT": "SHORT",
        }.get(signal)
        opens_new_exposure = bool(
            target_position
            and str(current_position or "NEUTRAL").upper() != target_position
        )

        # Deterministic risk gate: the LLM decided direction; position size is
        # recomputed mathematically for position-opening actions when enabled.
        effective_amount = dollar_amount
        risk_sizing_info = None
        risk_stop_price = None
        if risk_params is not None and opens_new_exposure:
            order_side = "sell" if signal == "SHORT" else "buy"
            try:
                sizing = AlpacaUtils.compute_risk_sized_amount(
                    symbol=symbol,
                    confidence=intent.confidence,
                    requested_notional=dollar_amount,
                    risk_params=risk_params,
                    side=order_side,
                )
            except Exception as e:
                return {
                    "success": False,
                    "broker_attempted": False,
                    "error": f"Risk sizing unavailable; new exposure blocked: {e}",
                    "trade_intent": intent.model_dump(mode="json"),
                    "intent_warnings": warnings,
                    "risk_sizing": {"applied": False, "error": str(e)},
                }
            else:
                if not sizing.approved:
                    return {
                        "success": False,
                        "error": f"Trade blocked by deterministic risk engine: {sizing.reason}",
                        "trade_intent": intent.model_dump(mode="json"),
                        "intent_warnings": warnings,
                        "risk_sizing": {"applied": True, **sizing.to_dict()},
                    }
                effective_amount = sizing.notional
                risk_stop_price = sizing.stop_loss_price
                risk_sizing_info = {"applied": True, **sizing.to_dict()}

        protective_prices = (
            AlpacaUtils._resolve_protective_prices(
                intent, signal, is_crypto, warnings
            )
            if opens_new_exposure
            else None
        )
        controls = intent.risk_controls
        if (
            opens_new_exposure and not is_crypto and protective_prices is None
            and get_config().get("protective_bracket_orders_enabled", True)
            and (controls.stop_loss_price is not None or controls.take_profit_price is not None)
        ):
            return {
                "success": False, "broker_attempted": False,
                "error": "Invalid protective prices; new exposure blocked.",
                "trade_intent": intent.model_dump(mode="json"),
                "intent_warnings": warnings,
                "protective_order_status": "invalid_controls",
            }
        if (
            risk_stop_price
            and not is_crypto
            and get_config().get("protective_bracket_orders_enabled", True)
            and not controls.stop_loss_price
        ):
            target_price = (
                (protective_prices or {}).get("take_profit_price")
                or controls.take_profit_price
            )
            stop_is_consistent = not target_price or (
                signal in {"BUY", "LONG"} and risk_stop_price < target_price
            ) or (signal == "SHORT" and target_price < risk_stop_price)
            if stop_is_consistent:
                protective_prices = dict(protective_prices or {})
                protective_prices["stop_loss_price"] = risk_stop_price

        execute_kwargs = dict(
            symbol=symbol,
            current_position=current_position,
            signal=signal,
            dollar_amount=effective_amount,
            allow_shorts=allow_shorts,
        )
        if protective_prices:
            execute_kwargs["protective_prices"] = protective_prices

        result = AlpacaUtils.execute_trading_action(**execute_kwargs)
        result["trade_intent"] = intent.model_dump(mode="json")

        protective_status = "advisory_only"
        for action in result.get("actions", []):
            action_result = action.get("result") or {}
            if action_result.get("order_class") == "bracket":
                protective_status = "submitted_bracket"
            elif action_result.get("order_class") == "oto":
                protective_status = "submitted_oto"
            elif action_result.get("protective_failed"):
                protective_status = "submission_failed"
                warnings.append(
                    "Protected entry could not be confirmed; no unprotected replacement was submitted."
                )
        if protective_status == "advisory_only" and opens_new_exposure and (
            intent.risk_controls.required_controls
            or intent.risk_controls.stop_loss
            or intent.risk_controls.take_profit
        ):
            warnings.append("Broker stop-loss/take-profit orders were not submitted; controls remain advisory.")

        result["intent_warnings"] = warnings
        result["protective_order_status"] = protective_status
        if risk_sizing_info is not None:
            result["risk_sizing"] = risk_sizing_info
        return result

    @staticmethod
    def _resolve_protective_prices(intent, signal, is_crypto, warnings):
        """Decide which protective price levels can be submitted to the broker.

        Returns a dict for place_protected_market_order, or None when the
        intent must stay advisory (no numeric levels, crypto asset, disabled
        by config, non-opening action, or inconsistent levels).
        """
        from tradingagents.agents.schemas import extract_protective_price

        controls = intent.risk_controls
        stop_price = controls.stop_loss_price or extract_protective_price(controls.stop_loss)
        target_price = controls.take_profit_price or extract_protective_price(controls.take_profit)
        if not stop_price and not target_price:
            return None

        opening_long = signal in {"BUY", "LONG"}
        opening_short = signal == "SHORT"
        if not opening_long and not opening_short:
            return None  # closes/holds carry nothing to protect

        if is_crypto:
            warnings.append(
                "Protective bracket/OTO orders are not supported for crypto assets; controls remain advisory."
            )
            return None

        if not get_config().get("protective_bracket_orders_enabled", True):
            warnings.append(
                "Protective bracket orders are disabled by configuration; controls remain advisory."
            )
            return None

        if stop_price and target_price:
            inverted = (opening_long and stop_price >= target_price) or (
                opening_short and target_price >= stop_price
            )
            if inverted:
                warnings.append(
                    f"Protective prices are inconsistent for a {'long' if opening_long else 'short'} entry "
                    f"(stop={stop_price}, target={target_price}); entry requires corrected levels."
                )
                return None

        return {"stop_loss_price": stop_price, "take_profit_price": target_price}

    @staticmethod
    def _safety_context(symbol: str):
        """Best-effort account snapshot for the safety layer's breakers.

        Returns (account, position_value); either may be None when the broker
        is unreachable. New exposure is then blocked until it can be verified.
        """
        try:
            client = get_alpaca_trading_client()
            acct = client.get_account()
            account = {
                "equity": float(acct.equity),
                "last_equity": float(acct.last_equity),
            }
        except Exception:
            return None, None

        position_value = 0.0
        try:
            key = symbol.upper().replace("/", "")
            for pos in client.get_all_positions():
                if pos.symbol.upper() == key:
                    position_value = abs(float(pos.market_value))
                    break
        except Exception:
            position_value = None
        return account, position_value

    @staticmethod
    def execute_trading_action(symbol: str, current_position: str, signal: str,
                             dollar_amount: float, allow_shorts: bool = False,
                             protective_prices: Optional[Dict[str, float]] = None) -> dict:
        """
        Execute trading action based on current position and signal
        
        Args:
            symbol: Stock symbol
            current_position: Current position state ("LONG", "SHORT", "NEUTRAL")
            signal: Trading signal from analysis
            dollar_amount: Dollar amount for trades
            allow_shorts: Whether short selling is allowed
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Deterministic safety gate — consulted before any broker call and
            # entirely independent of the agents' reasoning. Cheap local checks
            # (kill switch, notional cap, rejection streak) run first; account-
            # based circuit breakers run only if those pass.
            guard = None
            try:
                from tradingagents.safety import get_safety_guard

                guard = get_safety_guard()
            except Exception:
                guard = None

            results = []

            def _check_safety(
                sym: str, amount: float, *, risk_reducing: bool = False
            ):
                """Run the guard immediately before an actual broker order.

                Position flips close the old exposure first, then independently
                gate the new exposure. This prevents a tripped loss breaker from
                trapping a position while still refusing the replacement order.
                """
                from tradingagents.safety.guardrails import SafetyVerdict

                if guard is None and not risk_reducing:
                    return SafetyVerdict(False, ["Safety controls unavailable; new exposure blocked."])
                if guard is None or not guard.enabled:
                    return None
                verdict = guard.check_order(
                    sym,
                    amount,
                    risk_reducing=risk_reducing,
                )
                if verdict.allowed and not risk_reducing:
                    account_state, position_value = AlpacaUtils._safety_context(sym)
                    if (
                        not account_state or position_value is None
                        or not all(math.isfinite(float(account_state.get(key, float("nan"))))
                                   and float(account_state[key]) > 0 for key in ("equity", "last_equity"))
                        or not math.isfinite(float(position_value))
                    ):
                        return SafetyVerdict(False, ["Account risk data unavailable; new exposure blocked."])
                    else:
                        verdict = guard.check_order(
                            sym,
                            amount,
                            account=account_state,
                            position_value=position_value,
                        )
                return verdict

            def _safety_failure(sym: str, verdict) -> dict:
                error_msg = "Safety layer blocked order flow: " + " ".join(
                    verdict.reasons
                )
                print(f"[SAFETY] {error_msg}")
                # Ops alert (deduped by reason, so a tripped breaker alerts
                # once per cooldown window, not once per order).
                try:
                    from tradingagents.alerts import notify_safety_block

                    notify_safety_block(sym, verdict.reasons)
                except Exception:
                    pass
                return {
                    "success": False,
                    "safety_blocked": True,
                    "broker_attempted": False,
                    "error": error_msg,
                    "safety_checks": verdict.checks,
                }

            def _close_position(sym: str) -> dict:
                verdict = _check_safety(sym, 0.0, risk_reducing=True)
                if verdict is not None and not verdict.allowed:
                    return _safety_failure(sym, verdict)
                return AlpacaUtils.close_position(sym)

            # Helper to calculate integer quantity for any orders (used by both trading modes)
            def _calc_qty(sym: str, amount: float) -> Optional[int]:
                """Return integer share qty from the latest quote, or None when no
                trustworthy price exists. Never guess a price: a wrong assumption
                converts a dollar budget into that many shares."""
                try:
                    quote = AlpacaUtils.get_latest_quote(sym)
                    prices = [float(quote.get(key) or 0) for key in ("bid_price", "ask_price")]
                    price = max(prices)
                except Exception:
                    return None
                if not math.isfinite(price) or price <= 0 or not math.isfinite(amount) or amount <= 0:
                    return None
                qty = int(amount / price)
                return qty if qty >= 1 else None

            def _open_position(sym: str, side: str, amount: float) -> dict:
                """Open a position, attaching broker protective orders when available.

                Never retry an ambiguous protected submission as a plain order:
                the first request may already have reached the broker.
                """
                if not math.isfinite(amount) or amount <= 0:
                    return {"success": False, "broker_attempted": False,
                            "error": "New exposure blocked: no positive verified notional available."}
                verdict = _check_safety(sym, amount)
                if verdict is not None and not verdict.allowed:
                    return _safety_failure(sym, verdict)

                is_crypto_sym = "/" in sym.upper()
                if is_crypto_sym and side == "buy":
                    # Crypto buys use exact notional sizing (no bracket support).
                    return AlpacaUtils.place_market_order(sym, side, notional=amount)

                qty_int = _calc_qty(sym, amount)
                if qty_int is None:
                    return {
                        "success": False,
                        "broker_attempted": False,
                        "error": (
                            f"No trustworthy price or affordable whole-share quantity for {sym}; "
                            f"{side} order skipped"
                        ),
                    }
                if not is_crypto_sym and protective_prices:
                    protected = AlpacaUtils.place_protected_market_order(
                        sym,
                        side,
                        qty_int,
                        stop_loss_price=protective_prices.get("stop_loss_price"),
                        take_profit_price=protective_prices.get("take_profit_price"),
                    )
                    if protected.get("success"):
                        return protected
                    protected["protective_failed"] = True
                    return protected
                return AlpacaUtils.place_market_order(sym, side, qty=qty_int)

            pending_close = False

            def _close_filled(result: dict) -> bool:
                nonlocal pending_close
                if not result.get("success"):
                    return False
                status = getattr(result.get("status"), "value", result.get("status"))
                if status == "filled":
                    return True
                pending_close = True
                results.append({"action": "wait_for_close", "message":
                                "Close order is pending; reassess the live position before opening new exposure."})
                return False

            if allow_shorts:
                # Trading mode: LONG/NEUTRAL/SHORT signals
                signal = signal.upper()
                
                if current_position == "LONG":
                    if signal == "LONG":
                        results.append({"action": "hold", "message": f"Keeping LONG position in {symbol}"})
                    elif signal == "NEUTRAL":
                        # Close LONG position
                        close_result = _close_position(symbol)
                        results.append({"action": "close_long", "result": close_result})
                    elif signal == "SHORT":
                        # Close LONG and open SHORT
                        close_result = _close_position(symbol)
                        results.append({"action": "close_long", "result": close_result})
                        if _close_filled(close_result):
                            # Check if this is crypto - Alpaca doesn't support crypto short selling directly
                            is_crypto = "/" in symbol.upper()
                            if is_crypto:
                                error_msg = f"Direct short selling not supported for crypto assets like {symbol}. Position closed but short not opened."
                                results.append({"action": "open_short", "result": {"success": False, "error": error_msg}})
                            else:
                                short_result = _open_position(symbol, "sell", dollar_amount)
                                results.append({"action": "open_short", "result": short_result})
                
                elif current_position == "SHORT":
                    if signal == "SHORT":
                        results.append({"action": "hold", "message": f"Keeping SHORT position in {symbol}"})
                    elif signal == "NEUTRAL":
                        # Close SHORT position
                        close_result = _close_position(symbol)
                        results.append({"action": "close_short", "result": close_result})
                    elif signal == "LONG":
                        # Close SHORT and open LONG
                        close_result = _close_position(symbol)
                        results.append({"action": "close_short", "result": close_result})
                        if _close_filled(close_result):
                            long_result = _open_position(symbol, "buy", dollar_amount)
                            results.append({"action": "open_long", "result": long_result})
                
                elif current_position == "NEUTRAL":
                    if signal == "LONG":
                        long_result = _open_position(symbol, "buy", dollar_amount)
                        results.append({"action": "open_long", "result": long_result})
                    elif signal == "SHORT":
                        # Check if this is crypto - Alpaca doesn't support crypto short selling directly
                        is_crypto = "/" in symbol.upper()
                        if is_crypto:
                            error_msg = f"Direct short selling not supported for crypto assets like {symbol}. Consider using derivatives or margin trading platforms."
                            results.append({"action": "open_short", "result": {"success": False, "error": error_msg}})
                        else:
                            short_result = _open_position(symbol, "sell", dollar_amount)
                            results.append({"action": "open_short", "result": short_result})
                    elif signal == "NEUTRAL":
                        results.append({"action": "hold", "message": f"No position needed for {symbol}"})
            
            else:
                # Investment mode: BUY/HOLD/SELL signals
                signal = signal.upper()
                has_position = current_position == "LONG"
                
                if signal == "BUY":
                    if has_position:
                        results.append({"action": "hold", "message": f"Already have position in {symbol}"})
                    else:
                        buy_result = _open_position(symbol, "buy", dollar_amount)
                        results.append({"action": "buy", "result": buy_result})
                
                elif signal == "SELL":
                    if has_position:
                        # Sell position
                        sell_result = _close_position(symbol)
                        results.append({"action": "sell", "result": sell_result})
                    else:
                        results.append({"action": "hold", "message": f"No position to sell in {symbol}"})
                
                elif signal == "HOLD":
                    results.append({"action": "hold", "message": f"Holding current position in {symbol}"})

            if not results:
                return {
                    "success": False,
                    "symbol": symbol,
                    "current_position": current_position,
                    "signal": signal,
                    "actions": [],
                    "error": f"Unsupported trading signal '{signal}' for allow_shorts={allow_shorts}",
                }
            
            # Check if any critical actions failed
            has_failures = False
            for action in results:
                if "result" in action:
                    action_result = action["result"]
                    order_success = bool(action_result.get("success", True))
                    if (
                        guard is not None
                        and guard.enabled
                        and action_result.get("broker_attempted", True)
                    ):
                        # Feed the consecutive-rejection circuit breaker.
                        try:
                            guard.record_order_result(order_success)
                        except Exception:
                            pass
                    if not order_success:
                        has_failures = True

            response = {
                "success": not has_failures,
                "symbol": symbol,
                "current_position": current_position,
                "signal": signal,
                "actions": results
            }
            if pending_close:
                response["pending_close"] = True
            safety_failure = next(
                (
                    action["result"]
                    for action in results
                    if action.get("result", {}).get("safety_blocked")
                ),
                None,
            )
            if safety_failure:
                response.update(
                    safety_blocked=True,
                    error=safety_failure.get("error"),
                    safety_checks=safety_failure.get("safety_checks", {}),
                )
            return response
            
        except Exception as e:
            error_msg = f"Error executing trading action for {symbol}: {e}"
            print(error_msg)
            return {"success": False, "error": error_msg} 
