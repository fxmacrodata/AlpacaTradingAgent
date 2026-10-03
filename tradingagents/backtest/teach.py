"""Backtest -> self-learning-memory bridge ("intensive teaching").

Replays the decisions this deployment already recorded under
``eval_results/`` against historical prices and injects one dated lesson per
decision into the persistent per-agent ChromaDB memories, so a fresh agent
starts with the distilled experience of its own recorded history instead of
an empty memory (FinMem's train-then-test warm-up, arXiv:2311.13743).

Discipline mirrors the backtest engine: a decision made on day t enters at
the *next* bar's open and its outcome is measured at the open ``horizon_bars``
later. This is a hypothetical fixed-horizon return, not realized account
PnL: fills, holding state and costs are not reconstructed here. Only runs
completed on their analysis date are eligible; their analyst inputs still
need independent point-in-time validation.

Teaching is idempotent: every lesson carries a ``teach_key`` and memories
that already hold it are skipped, so re-running the bridge never duplicates.
Lessons are also tagged ``source=backtest_teach`` so later memory
maintenance can treat batch-taught lessons differently from live ones.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import pandas as pd

from .engine import normalize_price_frame
from .signals import load_recorded_runs, normalize_action

_REPORT_KEYS = (
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "macro_report",
)


def compute_decision_outcomes(
    prices: pd.DataFrame,
    signals: Dict[str, str],
    horizon_bars: int = 5,
) -> List[dict]:
    """Hypothetical directional outcome under next-open execution.

    As in the daily replay engine, a signal is consumed at the first
    available bar close on/after t; entry is the following bar's open.
    Exit is the open ``horizon_bars`` bars later (or the last
    bar, flagged ``partial``). ``decision_return`` is signed by the action:
    BUY earns the asset move and SHORT its negation. SELL, NEUTRAL and HOLD
    have no inferred return because position and cost-basis history are not
    available. Decisions with no bar after them are dropped.
    """
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be at least 1.")

    frame = normalize_price_frame(prices)
    bar_dates = [ts.date().isoformat() for ts in frame.index]
    opens = frame["open"].tolist()

    outcomes: List[dict] = []
    for trade_date, action in sorted((signals or {}).items()):
        action = normalize_action(action)
        if action is None:
            raise ValueError("Unrecognized action in recorded outcomes")
        signal_idx = next((i for i, d in enumerate(bar_dates) if d >= trade_date), None)
        if signal_idx is None or signal_idx + 1 >= len(frame):
            continue
        entry_idx = signal_idx + 1

        exit_idx = entry_idx + horizon_bars
        partial = exit_idx > len(frame) - 1
        if partial:
            exit_idx = len(frame) - 1
        if exit_idx <= entry_idx:
            continue

        entry_price = float(opens[entry_idx])
        exit_price = float(opens[exit_idx])
        if entry_price <= 0:
            continue
        asset_return = exit_price / entry_price - 1.0

        if action == "BUY":
            decision_return = asset_return
        elif action == "SHORT":
            decision_return = -asset_return
        else:
            decision_return = None

        outcomes.append(
            {
                "trade_date": trade_date,
                "action": action,
                "entry_date": bar_dates[entry_idx],
                "entry_price": entry_price,
                "exit_date": bar_dates[exit_idx],
                "exit_price": exit_price,
                "horizon_used": exit_idx - entry_idx,
                "asset_return": asset_return,
                "decision_return": decision_return,
                "partial": partial,
            }
        )
    return outcomes


def _situation_from_state(state: dict) -> Optional[str]:
    parts = [str(state[k]) for k in _REPORT_KEYS if state.get(k)]
    return "\n\n".join(parts) if parts else None


def _deterministic_lesson(symbol: str, outcome: dict) -> str:
    action = outcome["action"]
    horizon = outcome["horizon_used"]
    asset_pct = f"{outcome['asset_return']:+.1%}"
    span = "" if not outcome["partial"] else " (horizon truncated by data end)"

    if outcome["decision_return"] is None:
        body = (
            f"The asset moved {asset_pct} over the next {horizon} bars{span}. "
            f"The {action} action does not establish profit or loss without "
            "the held position, its cost basis and actual fills."
        )
    else:
        realized = f"{outcome['decision_return']:+.1%}"
        body = (
            f"The {action} direction had a hypothetical return of {realized} before costs over the next {horizon} "
            f"bars{span} (asset moved {asset_pct}; entry at the next open "
            f"{outcome['entry_price']:.4g} on {outcome['entry_date']}, exit at "
            f"the open {outcome['exit_price']:.4g} on {outcome['exit_date']})."
        )
    verdict = "One observation does not establish an edge; compare repeated out-of-sample results after costs."

    return (
        f"[Backtest lesson] {symbol} on {outcome['trade_date']}: final decision "
        f"was {action}. {body} {verdict}"
    )


def default_agent_memories(config: Optional[dict] = None) -> Dict[str, object]:
    """The five reflection memories, named exactly as TradingAgentsGraph names
    them so batch-taught lessons are what the live agents later retrieve.

    Imports are deferred: tradingagents.agents pulls dataflows at import
    time, and the backtest package must stay importable without it.
    """
    from tradingagents.agents.utils.memory import FinancialSituationMemory

    if config is None:
        from tradingagents.default_config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG
    return {
        name: FinancialSituationMemory(f"{name}_memory", config)
        for name in ("bull", "bear", "trader", "invest_judge", "risk_manager")
    }


def teach_memories_from_history(
    symbol: str,
    memories: Dict[str, object],
    price_loader: Optional[Callable[[str, str, Optional[str]], pd.DataFrame]] = None,
    reflector=None,
    horizon_bars: int = 5,
    eval_results_dir: str = "eval_results",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Batch-inject realized-outcome lessons for `symbol` into agent memories.

    `memories` maps component name -> FinancialSituationMemory (the same map
    reflect_on_outcome uses). When `reflector` is given, each decision's
    lesson is written by one quick-LLM call
    (``reflect_on_final_decision``); otherwise a deterministic template is
    used — zero LLM cost, embeddings only.

    Returns a summary dict; raises ValueError when the symbol has no
    recorded completed decisions to teach from.
    """
    runs = load_recorded_runs(symbol, eval_results_dir=eval_results_dir)
    signals = {day: normalize_action(run["summary"]["final_signal"]) for day, run in runs.items()}
    if start_date:
        signals = {d: a for d, a in signals.items() if d >= start_date}
    if end_date:
        signals = {d: a for d, a in signals.items() if d <= end_date}
    if not signals:
        raise ValueError(
            f"No recorded completed runs with final signals found for {symbol} "
            f"under {eval_results_dir}/."
        )

    if price_loader is None:
        from tradingagents.dataflows.alpaca_utils import AlpacaUtils

        price_loader = AlpacaUtils.get_stock_data

    prices = price_loader(symbol, min(signals), end_date)
    if end_date:
        prices = normalize_price_frame(prices)
        prices = prices[[ts.date().isoformat() <= end_date for ts in prices.index]]
    outcomes = compute_decision_outcomes(prices, signals, horizon_bars=horizon_bars)

    summary = {
        "symbol": symbol,
        "signals_found": len(signals),
        "outcomes_computed": len(outcomes),
        "decisions_taught": 0,
        "decisions_skipped_duplicate": 0,
        "decisions_skipped_no_state": 0,
        "lessons_written": 0,
    }

    for outcome in outcomes:
        trade_date = outcome["trade_date"]
        teach_key = f"{symbol}|{trade_date}|{horizon_bars}"

        targets = {
            name: memory
            for name, memory in memories.items()
            if memory is not None
            and not memory.has_metadata_value("teach_key", teach_key)
        }
        if not targets:
            summary["decisions_skipped_duplicate"] += 1
            continue

        # Use the exact same run that supplied the signal, never a later rerun.
        state = (runs[trade_date].get("snapshots") or {}).get("final_state")
        situation = _situation_from_state(state) if state else None
        if not situation:
            summary["decisions_skipped_no_state"] += 1
            continue

        if reflector is not None and outcome["decision_return"] is not None:
            final_decision = str(
                state.get("final_trade_decision") or outcome["action"]
            )
            final_decision += "\nEvaluation is a hypothetical fixed-horizon return before costs, not realized account PnL."
            lesson = reflector.reflect_on_final_decision(
                final_decision,
                raw_return=outcome["decision_return"],
                alpha_return=None,
            )
        else:
            lesson = _deterministic_lesson(symbol, outcome)

        metadata = {
            "source": "backtest_teach",
            "teach_key": teach_key,
            "symbol": symbol,
            "trade_date": trade_date,
            "action": outcome["action"],
            "horizon_bars": int(horizon_bars),
            "outcome_date": outcome["exit_date"],
            "return_basis": "hypothetical_before_costs",
        }
        if outcome["decision_return"] is not None:
            metadata["decision_return"] = float(outcome["decision_return"])
        written = 0
        for memory in targets.values():
            before = memory.situation_collection.count()
            memory.add_situations([(situation, lesson)], extra_metadata=metadata)
            written += memory.situation_collection.count() - before

        # written == 0 means embeddings were unavailable and nothing stored;
        # such decisions are simply not counted as taught.
        if written:
            summary["decisions_taught"] += 1
            summary["lessons_written"] += written

    return summary
