"""FXMacroData helpers for the macro analyst.

FRED only covers the US. These helpers add headline releases for other
economies (policy rate, CPI, unemployment, GDP, government bond yields), the
scheduled economic release calendar, and FX reference rates.

API docs: https://fxmacrodata.com/documentation/reference

USD announcements and the USD calendar are readable without a key (recent
90 days, delayed 15 minutes). Other currencies and FX rates need
FXMACRODATA_API_KEY.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests

from .config import get_config, get_fxmacrodata_api_key

BASE_URL = "https://api.fxmacrodata.com/v1"
REQUEST_TIMEOUT = 15
PAGE_LIMIT = 100  # API maximum per page
MAX_PAGES = 5

# Headline series shown per currency. Slugs follow /v1/data_catalogue/{currency};
# a currency that does not publish one of these is simply skipped.
HEADLINE_INDICATORS = {
    "policy_rate": "Policy Rate",
    "inflation": "Inflation",
    "core_inflation": "Core Inflation",
    "unemployment": "Unemployment Rate",
    "gdp": "GDP",
    "gov_bond_2y": "2Y Government Bond",
    "gov_bond_10y": "10Y Government Bond",
}

DEFAULT_CURRENCIES = ["USD", "EUR", "GBP", "JPY"]
DEFAULT_FX_PAIRS = ["EUR/USD", "USD/JPY", "GBP/USD"]


def _fxmacrodata_get(path: str, params: Optional[Dict] = None) -> Dict:
    """GET an FXMacroData endpoint. Returns the JSON body or {"error": ...}."""
    headers = {"Accept": "application/json"}
    api_key = get_fxmacrodata_api_key()
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        response = requests.get(
            f"{BASE_URL}{path}",
            params=params or {},
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch FXMacroData {path}: {str(e)}"}

    try:
        body = response.json()
    except ValueError:
        body = {}

    if response.status_code in (401, 403):
        return {
            "error": f"FXMacroData {path} returned HTTP {response.status_code}: a valid "
            "FXMACRODATA_API_KEY is required (USD works without a key).",
            "key_required": True,
        }
    if response.status_code >= 400:
        detail = body.get("detail") if isinstance(body, dict) else None
        return {"error": f"FXMacroData {path} returned HTTP {response.status_code}: {detail or response.reason}"}
    return body


def _fetch_rows(path: str, params: Dict) -> Dict:
    """Fetch every row of a paginated list endpoint (offset + pagination.has_more)."""
    rows: List[Dict] = []
    params = dict(params)
    params.setdefault("limit", PAGE_LIMIT)
    offset = 0
    body: Dict = {}

    for _ in range(MAX_PAGES):
        params["offset"] = offset
        body = _fxmacrodata_get(path, params)
        if "error" in body:
            return body
        page = body.get("data") or []
        rows.extend(page)
        pagination = body.get("pagination") or {}
        if not pagination.get("has_more") or not page:
            break
        offset = pagination.get("next_offset") or offset + len(page)

    body = dict(body)
    body["data"] = rows
    return body


def _parse_currencies(currencies) -> List[str]:
    if not currencies:
        currencies = get_config().get("fxmacrodata_currencies") or DEFAULT_CURRENCIES
    if isinstance(currencies, str):
        currencies = currencies.split(",")
    return [c.strip().upper() for c in currencies if c and c.strip()]


def _split_by_access(currencies: List[str]):
    """Without a key only USD is served, so don't spend requests on the rest."""
    if get_fxmacrodata_api_key():
        return currencies, []
    return [c for c in currencies if c == "USD"], [c for c in currencies if c != "USD"]


def _end_of_day_epoch(curr_date: str) -> int:
    day = datetime.strptime(curr_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int((day + timedelta(days=1)).timestamp())


def _released_rows(rows: List[Dict], curr_date: str) -> List[Dict]:
    """Keep rows that have a value and were public by the end of curr_date.

    `val` can be null for a scheduled-but-unpublished period; those rows are
    dropped rather than read as zero. The announcement time check keeps
    backtests from seeing a print before it was released.
    """
    cutoff = _end_of_day_epoch(curr_date)
    released = []
    for row in rows:
        if row.get("val") is None:
            continue
        announced = row.get("announcement_datetime")
        if announced is not None and announced >= cutoff:
            continue
        released.append(row)
    return released


def _format_value(value: float, unit: Optional[str]) -> str:
    if unit and unit.strip() == "%":
        return f"{value:.2f}%"
    if unit:
        return f"{value:,.2f} {unit}"
    return f"{value:,.2f}"


def get_global_macro_indicators(curr_date: str, currencies=None) -> str:
    """
    Latest headline macro releases per currency, as known on curr_date.

    Args:
        curr_date: Current date in YYYY-MM-DD format
        currencies: List or comma-separated string of currency codes
            (defaults to config "fxmacrodata_currencies")

    Returns:
        Markdown report with one table per currency
    """
    currency_list, skipped_for_key = _split_by_access(_parse_currencies(currencies))
    result = f"## Global Macro Indicators as of {curr_date} (FXMacroData)\n\n"

    for currency in currency_list:
        table_rows = []
        key_required = False

        for slug, label in HEADLINE_INDICATORS.items():
            data = _fxmacrodata_get(
                f"/announcements/{currency.lower()}/{slug}",
                {"end_date": curr_date, "limit": 6},
            )
            if "error" in data:
                if data.get("key_required"):
                    key_required = True
                    break
                continue

            rows = _released_rows(data.get("data") or [], curr_date)
            if not rows:
                continue

            unit = (data.get("value_metadata") or {}).get("source_unit")
            latest = rows[0]
            latest_value = float(latest["val"])
            previous_str = "-"
            change_str = "-"
            if len(rows) >= 2:
                previous_value = float(rows[1]["val"])
                previous_str = f"{_format_value(previous_value, unit)} ({rows[1]['date']})"
                change_str = f"{latest_value - previous_value:+.2f}"

            table_rows.append(
                f"| {data.get('name') or label} | {_format_value(latest_value, unit)} "
                f"| {latest['date']} | {previous_str} | {change_str} |\n"
            )

        if key_required:
            skipped_for_key.append(currency)
            continue

        result += f"### {currency}\n"
        if not table_rows:
            result += "No recent releases available.\n\n"
            continue
        result += "| Indicator | Latest | Period | Previous | Change |\n"
        result += "|-----------|--------|--------|----------|--------|\n"
        result += "".join(table_rows) + "\n"

    if skipped_for_key:
        result += (
            f"**Not loaded**: {', '.join(skipped_for_key)} need FXMACRODATA_API_KEY "
            "(only USD is available without a key).\n"
        )

    return result


def get_economic_release_calendar(curr_date: str, currencies=None, days_ahead: int = 10) -> str:
    """
    Scheduled economic releases from curr_date through curr_date + days_ahead.

    Args:
        curr_date: Current date in YYYY-MM-DD format
        currencies: List or comma-separated string of currency codes
        days_ahead: Number of days ahead to include (default 10)

    Returns:
        Markdown table of upcoming releases sorted by release time
    """
    currency_list, skipped_for_key = _split_by_access(_parse_currencies(currencies))
    end_date = (datetime.strptime(curr_date, "%Y-%m-%d") + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    result = f"## Economic Release Calendar ({curr_date} to {end_date}, FXMacroData)\n\n"

    events = []
    errors = []
    for currency in currency_list:
        data = _fxmacrodata_get(
            f"/calendar/{currency.lower()}",
            {"start_date": curr_date, "end_date": end_date},
        )
        if "error" in data:
            if data.get("key_required"):
                skipped_for_key.append(currency)
            else:
                errors.append(f"{currency}: {data['error']}")
            continue
        for row in data.get("data") or []:
            events.append((currency, row))

    events.sort(key=lambda item: item[1].get("announcement_datetime") or 0)

    if events:
        result += "| Date/Time (UTC) | Currency | Release | Importance | Reference Period |\n"
        result += "|-----------------|----------|---------|------------|------------------|\n"
        for currency, row in events:
            when = row.get("announcement_datetime_utc")
            if when:
                when = when.replace("T", " ")[:16]
            elif row.get("announcement_datetime"):
                when = datetime.fromtimestamp(row["announcement_datetime"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            else:
                when = row.get("date", "-")
            result += (
                f"| {when} | {currency} | {row.get('name') or row.get('release')} "
                f"| {row.get('event_importance') or '-'} | {row.get('reference_period') or '-'} |\n"
            )
    else:
        result += "No scheduled releases found for this window.\n"

    if skipped_for_key:
        result += (
            f"\n**Not loaded**: {', '.join(skipped_for_key)} need FXMACRODATA_API_KEY "
            "(only USD is available without a key).\n"
        )
    for error in errors:
        result += f"\n**Error**: {error}\n"

    return result


def get_fx_rates_report(curr_date: str, pairs=None, lookback_days: int = 30) -> str:
    """
    FX reference rates for a few pairs with change over the lookback window.
    Requires FXMACRODATA_API_KEY.

    Args:
        curr_date: Current date in YYYY-MM-DD format
        pairs: List or comma-separated string of pairs like "EUR/USD"
        lookback_days: Number of days to look back (default 30)

    Returns:
        Markdown table of FX rates
    """
    if not get_fxmacrodata_api_key():
        return "Error: FXMacroData FX rates require FXMACRODATA_API_KEY."

    if not pairs:
        pairs = get_config().get("fxmacrodata_fx_pairs") or DEFAULT_FX_PAIRS
    if isinstance(pairs, str):
        pairs = pairs.split(",")

    start_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    result = f"## FX Rates ({start_date} to {curr_date}, FXMacroData)\n\n"
    result += "| Pair | Latest | Date | Change | Range (Low - High) |\n"
    result += "|------|--------|------|--------|--------------------|\n"

    errors = []
    for pair in pairs:
        pair = pair.strip().upper().replace("-", "/")
        if "/" not in pair:
            continue
        base, quote = pair.split("/", 1)
        data = _fetch_rows(
            f"/forex/{base.lower()}/{quote.lower()}",
            {"start_date": start_date, "end_date": curr_date},
        )
        if "error" in data:
            errors.append(f"{pair}: {data['error']}")
            continue

        # Rows come back most recent first.
        rows = [row for row in data.get("data") or [] if row.get("val") is not None]
        if not rows:
            errors.append(f"{pair}: no data in window")
            continue

        latest = float(rows[0]["val"])
        oldest = float(rows[-1]["val"])
        values = [float(row["val"]) for row in rows]
        change_pct = (latest - oldest) / oldest * 100 if oldest else 0
        result += (
            f"| {pair} | {latest:.4f} | {rows[0]['date']} | {change_pct:+.2f}% "
            f"| {min(values):.4f} - {max(values):.4f} |\n"
        )

    for error in errors:
        result += f"\n**Error**: {error}\n"

    return result
