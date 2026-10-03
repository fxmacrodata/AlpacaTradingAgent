"""
Market hours utilities for validating trading hours and checking if the market is open.
"""

import datetime
import pytz
from typing import List, Tuple, Dict, Any

from functools import lru_cache

import exchange_calendars as xcals

# Whole-hour scheduling slots inside a regular NYSE session.
MARKET_OPEN_HOUR = 10
MARKET_CLOSE_HOUR = 15


@lru_cache(maxsize=8)
def _calendar(year: int):
    return xcals.get_calendar("XNYS", start=f"{year}-01-01", end=f"{year}-12-31")


def _get_eastern_timezone():
    return pytz.timezone("US/Eastern")


def _current_eastern_time() -> datetime.datetime:
    return datetime.datetime.now(pytz.utc).astimezone(_get_eastern_timezone())


def _coerce_to_eastern(target_datetime: datetime.datetime = None) -> datetime.datetime:
    eastern = _get_eastern_timezone()
    if target_datetime is None:
        return _current_eastern_time()
    if target_datetime.tzinfo is None:
        # Naive datetimes passed into this module are treated as Eastern wall clock time.
        return eastern.localize(target_datetime)
    return target_datetime.astimezone(eastern)

def validate_market_hours(hours_str: str) -> Tuple[bool, List[int], str]:
    """
    Validate market hours input string.
    
    Args:
        hours_str: String like "11" or "11,13" representing hours
        
    Returns:
        Tuple of (is_valid, parsed_hours_list, error_message)
    """
    if not hours_str or not hours_str.strip():
        return False, [], "Please enter at least one trading hour"
    
    try:
        # Parse comma-separated hours
        hours_parts = [h.strip() for h in hours_str.split(',') if h.strip()]
        if not hours_parts:
            return False, [], "Please enter at least one trading hour"
        
        hours = []
        for hour_str in hours_parts:
            hour = int(hour_str)
            if hour < MARKET_OPEN_HOUR or hour > MARKET_CLOSE_HOUR:
                return False, [], f"Hour {hour} is outside market hours ({MARKET_OPEN_HOUR}AM-{MARKET_CLOSE_HOUR}PM EST/EDT)"
            hours.append(hour)
        
        # Remove duplicates and sort
        hours = sorted(list(set(hours)))
        return True, hours, ""
        
    except ValueError:
        return False, [], "Please enter valid hour numbers (e.g., 11,13)"

def is_market_open(target_datetime: datetime.datetime = None) -> Tuple[bool, str]:
    """
    Check if the US stock market is open at the given datetime.
    
    Args:
        target_datetime: Datetime to check (defaults to current time)
        
    Returns:
        Tuple of (is_open, reason_if_closed)
    """
    target_datetime = _coerce_to_eastern(target_datetime)
    
    # Check if it's a weekend
    if target_datetime.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False, "Market is closed on weekends"
    
    calendar = _calendar(target_datetime.year)
    day = target_datetime.date().isoformat()
    if not calendar.is_session(day):
        return False, f"Market is closed for holiday on {day}"
    market_open = calendar.session_open(day).to_pydatetime()
    market_close = calendar.session_close(day).to_pydatetime()
    if target_datetime < market_open:
        return False, f"Market opens at {market_open.astimezone(_get_eastern_timezone()):%I:%M %p %Z}"
    if target_datetime >= market_close:
        return False, f"Market closed at {market_close.astimezone(_get_eastern_timezone()):%I:%M %p %Z}"

    return True, "Market is open"

def get_next_market_datetime(target_hour: int, from_datetime: datetime.datetime = None) -> datetime.datetime:
    """
    Get the next market datetime for the specified hour.
    
    Args:
        target_hour: Hour to target (e.g., 11 for 11 AM)
        from_datetime: Starting datetime (defaults to current time)
        
    Returns:
        Next datetime when market will be open at the target hour
    """
    from_datetime = _coerce_to_eastern(from_datetime)

    if not MARKET_OPEN_HOUR <= target_hour <= MARKET_CLOSE_HOUR:
        raise ValueError("Choose a whole hour from 10 through 15 Eastern, within the regular session.")
    eastern = _get_eastern_timezone()
    # Enumerate actual sessions, including early closes, and localize each
    # date separately so crossing DST does not retain yesterday's UTC offset.
    for year in (from_datetime.year, from_datetime.year + 1):
        for session in _calendar(year).sessions:
            if session.date() < from_datetime.date():
                continue
            target_dt = eastern.localize(datetime.datetime.combine(session.date(), datetime.time(target_hour)))
            if target_dt > from_datetime and is_market_open(target_dt)[0]:
                return target_dt
    raise ValueError("No valid market session found for the requested hour.")


def format_market_hours_info(hours: List[int]) -> Dict[str, Any]:
    """
    Format market hours information for display.
    
    Args:
        hours: List of hours (e.g., [11, 13])
        
    Returns:
        Dictionary with formatted information
    """
    if not hours:
        return {"error": "No hours provided"}
    
    # Format hours for display
    hours = sorted(set(hours))
    formatted_hours = []
    for hour in sorted(hours):
        if hour == 0:
            formatted_hours.append("12:00 AM")
        elif hour < 12:
            formatted_hours.append(f"{hour}:00 AM")
        elif hour == 12:
            formatted_hours.append("12:00 PM")
        else:
            formatted_hours.append(f"{hour-12}:00 PM")
    
    hours_str = " and ".join(formatted_hours)
    
    # Calculate next execution times
    next_executions = []
    for hour in hours:
        next_dt = get_next_market_datetime(hour)
        next_executions.append({
            "hour": hour,
            "formatted_hour": formatted_hours[hours.index(hour)],
            "next_datetime": next_dt,
            "next_formatted": next_dt.strftime("%A, %B %d at %I:%M %p %Z")
        })
    
    return {
        "hours": hours,
        "formatted_hours": hours_str,
        "next_executions": next_executions,
        "market_timezone": "US/Eastern"
    }
