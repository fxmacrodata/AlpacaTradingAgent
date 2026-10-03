"""Regression cases found while auditing execution and historical evaluation."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.backtest.engine import normalize_price_frame, run_backtest, run_walk_forward
from tradingagents.backtest.signals import load_recorded_signals
from tradingagents.dataflows.alpaca_utils import AlpacaUtils
from tradingagents.dataflows.macro_utils import get_fred_data, get_fed_calendar_and_minutes
from tradingagents.regime import regime_report_block


def bars(days=100):
    return pd.DataFrame({
        "open": [100.0] * days, "high": [102.0] * days,
        "low": [98.0] * days, "close": [101.0] * days,
        "volume": [1000.0] * days,
    }, index=pd.bdate_range("2025-01-01", periods=days))


def test_partial_close_uses_broker_percentage_units():
    client = MagicMock()
    client.close_position.return_value = SimpleNamespace(
        id="close", symbol="AAPL", side="sell", qty="50", status="accepted"
    )
    with patch("tradingagents.dataflows.alpaca_utils.get_alpaca_trading_client", return_value=client):
        result = AlpacaUtils.close_position("AAPL", percentage=50)
    assert result["success"]
    assert float(client.close_position.call_args.args[1].percentage) == 50


@pytest.mark.parametrize("percentage", [0, -1, 101, float("nan"), float("inf")])
def test_invalid_close_percentage_never_reaches_broker(percentage):
    with patch("tradingagents.dataflows.alpaca_utils.get_alpaca_trading_client") as client:
        assert not AlpacaUtils.close_position("AAPL", percentage)["success"]
    client.assert_not_called()


def test_buy_quantity_respects_ask_price():
    guard = SimpleNamespace(enabled=False)
    with patch("tradingagents.safety.get_safety_guard", return_value=guard), patch.object(
        AlpacaUtils, "get_latest_quote", return_value={"bid_price": 90, "ask_price": 110}
    ), patch.object(AlpacaUtils, "place_market_order", return_value={"success": True}) as submit:
        result = AlpacaUtils.execute_trading_action("AAPL", "NEUTRAL", "BUY", 1000)
    assert result["success"]
    assert submit.call_args.kwargs["qty"] == 9


def test_unfilled_close_does_not_immediately_open_reverse_position():
    with patch("tradingagents.safety.get_safety_guard", return_value=SimpleNamespace(enabled=False)), patch.object(
        AlpacaUtils, "close_position", return_value={"success": True, "order_id": "exit", "status": "accepted"}
    ), patch.object(AlpacaUtils, "place_market_order") as submit:
        result = AlpacaUtils.execute_trading_action("AAPL", "LONG", "SHORT", 1000, allow_shorts=True)
    submit.assert_not_called()
    assert result["pending_close"]


def test_regime_uses_analysis_cutoff_even_when_loader_returns_future_bars():
    history = bars()
    cutoff = "2025-03-03"
    with patch("tradingagents.regime.classify_regime") as classify:
        classify.return_value = SimpleNamespace(label="unknown")
        loader = MagicMock(return_value=history)
        regime_report_block("AAPL", price_loader=loader, as_of_date=cutoff)
    assert loader.call_args.args[2] == cutoff
    assert classify.call_args.args[0].index.max() <= pd.Timestamp(cutoff)


def test_fred_requests_the_historical_vintage_with_timeout():
    with patch("tradingagents.dataflows.macro_utils.get_fred_api_key", return_value="test"), patch(
        "tradingagents.dataflows.macro_utils.requests.get"
    ) as get:
        get.return_value.json.return_value = {"observations": []}
        get_fred_data("CPIAUCSL", "2020-01-01", "2020-06-01")
    kwargs = get.call_args.kwargs
    assert kwargs["params"]["realtime_start"] == "2020-06-01"
    assert kwargs["params"]["realtime_end"] == "2020-06-01"
    assert kwargs["timeout"] > 0


def test_macro_report_does_not_invent_an_outdated_meeting_schedule():
    with patch("tradingagents.dataflows.macro_utils.get_fred_data", return_value={"observations": []}):
        report = get_fed_calendar_and_minutes("2026-10-03")
    assert "2024 FOMC" not in report
    assert "Quantitative tightening operations" not in report


def test_historical_rerun_cannot_overwrite_a_contemporaneous_decision(tmp_path):
    root = tmp_path / "AAPL" / "TradingAgentsStrategy_logs" / "runs"
    root.mkdir(parents=True)
    for name, when, action in [("original", "2025-01-02", "BUY"), ("rerun", "2026-01-02", "SELL")]:
        (root / f"{name}.json").write_text(json.dumps({
            "status": "completed", "trade_date": "2025-01-02",
            "started_at": f"{when}T15:00:00+00:00",
            "ended_at": f"{when}T16:00:00+00:00", "summary": {"final_signal": action},
        }), encoding="utf-8")
    with pytest.warns(UserWarning, match="Excluded"):
        assert load_recorded_signals("AAPL", str(tmp_path)) == {"2025-01-02": "BUY"}


def test_neutral_exits_instead_of_keeping_a_long_position():
    result = run_backtest(bars(6), {"2025-01-01": "BUY", "2025-01-03": "NEUTRAL"}, slippage_model="none")
    assert [order["side"] for order in result.orders] == ["buy", "sell"]


def test_walk_forward_does_not_replay_an_old_buy_in_every_fold():
    result = run_walk_forward(bars(12), {"2025-01-01": "BUY"}, window_bars=6, min_window_bars=2)
    assert result.windows[1]["metrics"]["cumulative_return"] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0])
def test_backtest_rejects_unusable_prices(value):
    prices = bars(3)
    prices.iloc[1, prices.columns.get_loc("open")] = value
    with pytest.raises(ValueError):
        normalize_price_frame(prices)


def test_repeated_buy_does_not_rebalance_an_existing_position():
    prices = bars(6)
    prices.iloc[2:, :] = prices.iloc[2:, :] * 1.5
    result = run_backtest(prices, {"2025-01-01": "BUY", "2025-01-03": "BUY"})
    assert len(result.orders) == 1


def test_sell_never_opens_a_short_even_if_shorting_is_enabled():
    result = run_backtest(bars(6), {"2025-01-01": "SELL"}, allow_shorts=True)
    assert result.orders == []


@pytest.mark.parametrize("ended_at", [None, "2025-01-02T16:00:00", "invalid", "2025-01-01T12:00:00Z"])
def test_unknown_or_inconsistent_signal_availability_is_excluded(tmp_path, ended_at):
    root = tmp_path / "AAPL" / "TradingAgentsStrategy_logs" / "runs"
    root.mkdir(parents=True)
    (root / "run.json").write_text(json.dumps({
        "status": "completed", "trade_date": "2025-01-02", "started_at": "2025-01-02T15:00:00Z",
        "ended_at": ended_at, "summary": {"final_signal": "BUY"},
    }), encoding="utf-8")
    with pytest.warns(UserWarning, match="Excluded"):
        assert load_recorded_signals("AAPL", str(tmp_path)) == {}


def test_equity_signal_uses_new_york_date_for_utc_completion(tmp_path):
    root = tmp_path / "AAPL" / "TradingAgentsStrategy_logs" / "runs"
    root.mkdir(parents=True)
    (root / "run.json").write_text(json.dumps({
        "status": "completed", "trade_date": "2025-01-02", "started_at": "2025-01-02T22:00:00Z",
        "ended_at": "2025-01-03T01:00:00Z", "summary": {"final_signal": "BUY"},
    }), encoding="utf-8")
    assert load_recorded_signals("AAPL", str(tmp_path)) == {"2025-01-02": "BUY"}


def test_sell_and_hold_lessons_do_not_invent_pnl_or_encourage_chasing():
    from tradingagents.backtest.teach import compute_decision_outcomes, _deterministic_lesson
    outcomes = compute_decision_outcomes(bars(), {"2025-01-01": "SELL", "2025-01-02": "HOLD"})
    for outcome in outcomes:
        assert outcome["decision_return"] is None
        lesson = _deterministic_lesson("AAPL", outcome)
        assert "does not establish profit or loss" in lesson
        assert "tradable signal" not in lesson


def test_memory_created_today_is_unavailable_to_historical_analysis():
    import uuid
    from tradingagents.agents.utils.memory import FinancialSituationMemory
    memory = FinancialSituationMemory(f"audit_{uuid.uuid4().hex}")
    memory.embeddings_enabled = True
    memory.get_embedding = lambda text: [0.1] * 8
    memory.add_situations([("future event", "future lesson")], extra_metadata={"created_at": "2020-01-01"})
    assert memory.get_memories("future event", as_of_date="2020-01-01") == []
    assert memory.get_memories("future event")[0]["recommendation"] == "future lesson"


def test_exchange_calendar_handles_early_close_future_holiday_and_dst():
    from datetime import datetime
    from webui.utils.market_hours import is_market_open, get_next_market_datetime
    assert not is_market_open(datetime(2026, 11, 27, 14))[0]
    assert not is_market_open(datetime(2028, 7, 4, 11))[0]
    assert not is_market_open(datetime(2026, 1, 5, 16))[0]
    following = get_next_market_datetime(11, datetime(2026, 3, 6, 12))
    assert following.strftime("%Y-%m-%d %H:%M %Z") == "2026-03-09 11:00 EDT"
    assert get_next_market_datetime(14, datetime(2026, 11, 26, 12)).date().isoformat() == "2026-11-30"
    with pytest.raises(ValueError):
        get_next_market_datetime(9)


def test_regime_all_configuration_fields_can_be_overridden():
    from tradingagents.regime import RegimeConfig, classify_regime
    cfg = RegimeConfig.from_config({"regime_vol_percentile_window": 60, "regime_calm_percentile": 20})
    assert cfg.vol_percentile_window == 60
    assert cfg.calm_percentile == 20
    prices = bars()
    prices["close"] = [100 + i % 5 for i in range(len(prices))]
    stock = classify_regime(prices, symbol="AAPL")
    crypto = classify_regime(prices, symbol="BTC/USD")
    assert crypto.metrics["annualized_vol_pct"] / stock.metrics["annualized_vol_pct"] == pytest.approx((365/252)**0.5)


@pytest.mark.parametrize("protected", [False, True])
def test_pending_entry_prevents_duplicate_orders(protected):
    client = MagicMock()
    client.get_orders.return_value = [SimpleNamespace(id="pending-entry")]
    with patch("tradingagents.dataflows.alpaca_utils.get_alpaca_trading_client", return_value=client):
        result = (AlpacaUtils.place_protected_market_order("AAPL", "buy", 5, stop_loss_price=90)
                  if protected else AlpacaUtils.place_market_order("AAPL", "buy", qty=5))
    assert not result["success"]
    assert not result["broker_attempted"]
    client.submit_order.assert_not_called()


def test_pending_order_lookup_failure_blocks_entry():
    client = MagicMock()
    client.get_orders.side_effect = ConnectionError("unavailable")
    with patch("tradingagents.dataflows.alpaca_utils.get_alpaca_trading_client", return_value=client):
        result = AlpacaUtils.place_market_order("AAPL", "buy", qty=5)
    assert not result["success"]
    client.submit_order.assert_not_called()


@pytest.mark.parametrize("equity", [None, 0, float("nan"), float("inf")])
def test_unknown_portfolio_equity_does_not_disable_exposure_cap(equity):
    from tradingagents.portfolio import assess_new_position
    verdict = assess_new_position("AAPL", 1000, equity, {}, {})
    assert not verdict.allowed
    assert verdict.adjusted_notional == 0


def test_short_entry_respects_portfolio_cap():
    from tradingagents.portfolio import adjust_new_position_notional
    amount = adjust_new_position_notional("AAPL", "SHORT", 1000, lambda: (10000, {"MSFT": -10000}, {}))
    assert amount == 0


def test_opposite_exposures_are_not_penalized_as_duplicate_risk():
    from tradingagents.portfolio import assess_new_position, PortfolioLimitsConfig
    prices = bars()
    prices["close"] = [100 + i % 5 for i in range(len(prices))]
    verdict = assess_new_position("AAPL", 1000, 100000, {"MSFT": -1000},
                                  {"AAPL": prices, "MSFT": prices},
                                  PortfolioLimitsConfig(vol_sizing_enabled=False))
    assert verdict.correlations["MSFT"] == pytest.approx(-1)
    assert verdict.adjusted_notional == 1000


def test_crypto_portfolio_history_uses_crypto_market_data_symbol():
    from tradingagents.portfolio import gather_portfolio_state_via_alpaca
    client = MagicMock()
    client.get_account.return_value = SimpleNamespace(equity="100000")
    client.get_all_positions.return_value = [SimpleNamespace(symbol="BTCUSD", market_value="1000", asset_class="crypto")]
    with patch("tradingagents.dataflows.alpaca_utils.get_alpaca_trading_client", return_value=client), patch.object(
        AlpacaUtils, "get_stock_data", return_value=bars()
    ) as load:
        _, positions, _ = gather_portfolio_state_via_alpaca("BTC/USD", lookback_bars=500)
    assert positions == {"BTCUSD": 1000}
    assert load.call_args.args[0] == "BTC/USD"
    from datetime import date
    assert (date.today() - date.fromisoformat(load.call_args.args[1])).days > 500


@pytest.mark.parametrize("exposure", [float("nan"), float("inf"), -100])
def test_invalid_exposure_cannot_be_used_for_risk_sizing(exposure):
    from tradingagents.risk.position_sizing import PositionSizer
    verdict = PositionSizer().size_position(equity=100000, price=100, atr=2, confidence="high",
                                            requested_notional=1000, current_gross_exposure=exposure)
    assert not verdict.approved


def test_reflection_uses_fixed_forward_horizon_and_analysis_cutoff():
    from datetime import date
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    prices = pd.DataFrame({"Open": [100, 110, 120, 130, 999]}, index=pd.bdate_range("2025-01-06", periods=5))
    with patch("tradingagents.graph.trading_graph.yf.download", return_value=prices):
        result = graph._fetch_return("AAPL", date(2025, 1, 3), 2, as_of_date=date(2025, 1, 8))
        assert result == pytest.approx(0.2)
        assert graph._fetch_return("AAPL", date(2025, 1, 3), 2, as_of_date=date(2025, 1, 7)) is None


def test_legacy_decision_log_cannot_leak_later_outcomes_into_history(tmp_path):
    from tradingagents.agents.utils.memory import TradingMemoryLog
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "memory.txt")})
    with patch.object(log, "load_entries", return_value=[{"pending": False, "ticker": "AAPL"}]):
        assert log.get_past_context("AAPL", as_of_date="2020-01-01") == ""


def test_zero_new_exposure_budget_still_allows_a_flip_to_close():
    with patch("tradingagents.safety.get_safety_guard", return_value=SimpleNamespace(enabled=False)), patch.object(
        AlpacaUtils, "close_position", return_value={"success": True, "status": "filled"}
    ) as close, patch.object(AlpacaUtils, "place_market_order") as submit:
        result = AlpacaUtils.execute_trading_action("AAPL", "SHORT", "LONG", 0, allow_shorts=True)
    close.assert_called_once()
    submit.assert_not_called()
    assert not result["success"]


def test_earnings_api_key_lookup_uses_runtime_configuration():
    from tradingagents.dataflows.earnings_utils import get_earnings_calendar_api_key
    with patch("tradingagents.dataflows.earnings_utils.get_api_key", return_value="test") as key:
        assert get_earnings_calendar_api_key() == "test"
    key.assert_called_once_with("earnings_calendar_api_key", "EARNINGS_CALENDAR_API_KEY")


def test_stock_symbol_containing_crypto_substring_keeps_earnings_data():
    from tradingagents.dataflows.earnings_utils import get_earnings_calendar_data
    with patch("tradingagents.dataflows.earnings_utils.get_finnhub_earnings_calendar", return_value="equity earnings") as earnings:
        assert get_earnings_calendar_data("ADAP", "2025-01-01", "2025-03-01") == "equity earnings"
    earnings.assert_called_once()


def test_weekend_teaching_entry_matches_the_daily_replay_engine():
    from tradingagents.backtest.teach import compute_decision_outcomes
    prices = bars(12)
    signals = {"2025-01-04": "BUY"}
    replay = run_backtest(prices, signals, slippage_model="none")
    outcome = compute_decision_outcomes(prices, signals, horizon_bars=2)[0]
    assert outcome["entry_date"] == replay.orders[0]["date"]
