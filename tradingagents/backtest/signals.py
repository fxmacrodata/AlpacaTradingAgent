"""Signal sources for backtesting.

The primary source replays decisions the multi-agent pipeline already made
and persisted under ``eval_results/`` (see tradingagents/run_logger.py), so
evaluating past agent behavior costs zero LLM calls. Signals are normalized
without losing the distinction between exiting (SELL/NEUTRAL), holding
(HOLD), and opening short exposure (SHORT).
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from zoneinfo import ZoneInfo

ACTION_ALIASES = {
    "BUY": "BUY",
    "LONG": "BUY",
    "SELL": "SELL",
    "SHORT": "SHORT",
    "HOLD": "HOLD",
    "NEUTRAL": "NEUTRAL",
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalize_action(raw) -> Optional[str]:
    """Normalize spelling while preserving exit and short-entry semantics."""
    if raw is None:
        return None
    return ACTION_ALIASES.get(str(raw).strip().upper())


def _sanitize_symbol_for_path(symbol: str) -> str:
    # Must mirror run_logger._sanitize_for_path so we find its output dirs
    # (importing it would trigger the module's stale-log recovery side effect).
    sanitized = re.sub(r"[^\w\-.]+", "_", symbol.strip())
    return sanitized or "unknown"


def load_recorded_runs(
    symbol: str,
    eval_results_dir: str = "eval_results",
) -> Dict[str, dict]:
    """Return completed runs that were available on their analysis date.

    A historical rerun made later cannot replace a contemporaneous decision.
    Completion timestamps must be timezone-aware; legacy logs without them
    cannot establish availability and are excluded. Daily equity replay uses
    New York session dates; crypto uses UTC. These checks establish signal
    timing, not point-in-time provenance of every underlying analyst input.
    """
    runs_dir = (
        Path(eval_results_dir)
        / _sanitize_symbol_for_path(symbol)
        / "TradingAgentsStrategy_logs"
        / "runs"
    )
    if not runs_dir.is_dir():
        return {}

    session_tz = ZoneInfo("UTC" if "/" in symbol else "America/New_York")
    best_per_date: Dict[str, tuple] = {}
    unavailable = 0
    for path in sorted(runs_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(payload, dict) or payload.get("status") != "completed":
            continue
        trade_date = str(payload.get("trade_date") or "").strip()
        if not _DATE_RE.match(trade_date):
            continue
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            continue
        action = normalize_action(summary.get("final_signal"))
        if action is None:
            continue
        try:
            analysis_date = date.fromisoformat(trade_date)
            started = datetime.fromisoformat(payload["started_at"].replace("Z", "+00:00"))
            ended = datetime.fromisoformat(payload["ended_at"].replace("Z", "+00:00"))
            if started.tzinfo is None or ended.tzinfo is None or ended < started:
                raise ValueError("Missing or inconsistent timezone-aware timestamps")
            if ended.astimezone(session_tz).date() != analysis_date:
                raise ValueError("Decision was not completed on its analysis date")
        except (KeyError, ValueError, TypeError, AttributeError):
            unavailable += 1
            continue
        current = best_per_date.get(trade_date)
        available_at = ended.astimezone(timezone.utc)
        if current is None or available_at > current[0]:
            best_per_date[trade_date] = (available_at, payload)

    if unavailable:
        warnings.warn(
            f"Excluded {unavailable} recorded run(s) for {symbol}: completion time "
            "is missing, invalid, or outside the analysis date. Historical reruns "
            "cannot establish out-of-sample performance.",
            UserWarning, stacklevel=2,
        )
    return {day: payload for day, (_, payload) in sorted(best_per_date.items())}


def load_recorded_signals(symbol: str, eval_results_dir: str = "eval_results") -> Dict[str, str]:
    """Read executable daily signals from contemporaneously completed runs."""
    return {
        day: normalize_action(payload["summary"]["final_signal"])
        for day, payload in load_recorded_runs(symbol, eval_results_dir).items()
    }
