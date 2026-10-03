"""Regression coverage for indicator requests observed during a live run."""

from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.dataflows.interface import get_stockstats_indicator_history
from tradingagents.dataflows.technical_brief import compute_indicators


@pytest.mark.parametrize("indicators,columns", [
    ("rsi_14,stochrsi_14,macd", ["rsi_14", "stoch_k", "stoch_d", "macd"]),
    ("close_8_ema,close_21_ema,close_50_sma,close_200_sma", ["ema_8", "ema_21", "sma_50", "sma_200"]),
    ("atr_14,bbands", ["atr_14", "boll_ub", "boll_lb"]),
    ("rsi, rsi_14", ["rsi_14"]),
])
def test_indicator_batches_return_all_requested_columns_once(indicators, columns):
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-03"], utc=True)})
    for col in columns:
        frame[col] = [40.0, 42.0, 99.0]
    with patch("tradingagents.dataflows.technical_brief.compute_indicators", return_value=frame) as compute:
        result = get_stockstats_indicator_history("NVDA", indicators, "2026-10-02")
    compute.assert_called_once()
    assert f"Requested indicator(s): {', '.join(columns)}" in result
    assert "2026-10-03" not in result
    for col in columns:
        assert f"- {col}: 42." in result


def test_unknown_indicator_is_not_silently_ignored():
    frame = pd.DataFrame({"timestamp": [pd.Timestamp("2026-10-02", tz="UTC")], "rsi_14": [50.0]})
    with patch("tradingagents.dataflows.technical_brief.compute_indicators", return_value=frame):
        result = get_stockstats_indicator_history("NVDA", "rsi_14,invented", "2026-10-02")
    assert "Error: unsupported indicator 'invented'" in result


def test_daily_history_warms_up_sma200_and_respects_configured_lookback():
    def price_loader(**kwargs):
        dates = pd.bdate_range(kwargs["start_date"], kwargs["end_date"])
        return pd.DataFrame({
            "timestamp": dates, "open": 100.0, "high": 102.0,
            "low": 98.0, "close": 101.0, "volume": 1000.0,
        })
    with patch("tradingagents.dataflows.technical_brief.AlpacaUtils.get_stock_data", side_effect=price_loader):
        frame = compute_indicators("NVDA", "2026-10-02", "1d")
        assert frame["sma_200"].iloc[-1] == 101.0
        with patch("tradingagents.dataflows.technical_brief.get_config", return_value={"technical_lookback_days": {"1d": 50}}):
            short_frame = compute_indicators("NVDA", "2026-10-02", "1d")
        assert pd.isna(short_frame["sma_200"].iloc[-1])
