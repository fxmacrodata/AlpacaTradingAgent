"""Tests for the deterministic portfolio-intelligence layer.

The layer sits above the per-symbol agent decisions and adjusts NEW
position sizes only: positive correlation with existing positions above a
threshold scales the size down, realized volatility above the target
scales it down (inverse-volatility / simplified risk parity), and a gross
exposure cap clips what remains. Factors never size a trade UP, missing
data never punishes, and only an explicit exposure clip may zero a trade.
"""

import unittest

import numpy as np
import pandas as pd

from tradingagents.portfolio import (
    PortfolioLimitsConfig,
    adjust_new_position_notional,
    assess_new_position,
    daily_returns,
    realized_daily_vol,
)


def _frame(closes, start="2025-01-06"):
    dates = pd.bdate_range(start=start, periods=len(closes))
    closes = list(closes)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
        }
    )


def _trending(n=80, drift=0.01, noise_seed=7, scale=0.001):
    rng = np.random.default_rng(noise_seed)
    returns = drift + rng.normal(0, scale, n)
    return list(100 * np.cumprod(1 + returns))


def _config(**overrides):
    return PortfolioLimitsConfig(**overrides)


class ReturnMathTests(unittest.TestCase):
    def test_daily_returns_shape(self):
        returns = daily_returns(_frame([100, 110, 99]))
        self.assertEqual(len(returns), 2)
        self.assertAlmostEqual(returns.iloc[0], 0.10)

    def test_realized_vol_of_constant_series_is_zero(self):
        vol = realized_daily_vol(daily_returns(_frame([100] * 30)))
        self.assertEqual(vol, 0.0)


class CorrelationPenaltyTests(unittest.TestCase):
    def test_highly_correlated_candidate_is_scaled_down(self):
        closes = _trending()
        # Same series shifted in level: correlation ~1.
        verdict = assess_new_position(
            symbol="MSFT",
            requested_notional=10_000.0,
            equity=100_000.0,
            open_positions={"AAPL": 20_000.0},
            price_history={"MSFT": _frame(closes), "AAPL": _frame([c * 2 for c in closes])},
            config=_config(vol_sizing_enabled=False),
        )
        self.assertLess(verdict.adjusted_notional, 10_000.0)
        self.assertAlmostEqual(
            verdict.adjusted_notional, 10_000.0 * verdict.config.correlated_size_factor
        )
        self.assertTrue(any("correlation" in r.lower() for r in verdict.reasons))

    def test_uncorrelated_candidate_keeps_full_size(self):
        rng = np.random.default_rng(3)
        a = list(100 * np.cumprod(1 + rng.normal(0, 0.01, 80)))
        b = list(100 * np.cumprod(1 + rng.normal(0, 0.01, 80)))
        verdict = assess_new_position(
            symbol="GLD",
            requested_notional=10_000.0,
            equity=100_000.0,
            open_positions={"AAPL": 20_000.0},
            price_history={"GLD": _frame(a), "AAPL": _frame(b)},
            config=_config(vol_sizing_enabled=False),
        )
        self.assertEqual(verdict.adjusted_notional, 10_000.0)

    def test_negative_correlation_is_not_punished(self):
        closes = _trending()
        inverse = [200 - c + 100 for c in closes]  # strongly negative corr
        verdict = assess_new_position(
            symbol="SH",
            requested_notional=10_000.0,
            equity=100_000.0,
            open_positions={"AAPL": 20_000.0},
            price_history={"SH": _frame(inverse), "AAPL": _frame(closes)},
            config=_config(vol_sizing_enabled=False),
        )
        self.assertEqual(verdict.adjusted_notional, 10_000.0)


class VolatilitySizingTests(unittest.TestCase):
    def test_high_vol_candidate_is_scaled_down(self):
        rng = np.random.default_rng(11)
        wild = list(100 * np.cumprod(1 + rng.normal(0, 0.06, 80)))  # ~6% daily vol
        verdict = assess_new_position(
            symbol="MEME",
            requested_notional=10_000.0,
            equity=100_000.0,
            open_positions={},
            price_history={"MEME": _frame(wild)},
            config=_config(target_daily_vol_pct=2.0),
        )
        self.assertLess(verdict.adjusted_notional, 10_000.0)
        self.assertTrue(any("volatility" in r.lower() for r in verdict.reasons))

    def test_calm_candidate_is_never_sized_up(self):
        calm = list(np.linspace(100, 101, 80))  # tiny vol
        verdict = assess_new_position(
            symbol="BOND",
            requested_notional=10_000.0,
            equity=100_000.0,
            open_positions={},
            price_history={"BOND": _frame(calm)},
            config=_config(target_daily_vol_pct=2.0),
        )
        self.assertEqual(verdict.adjusted_notional, 10_000.0)


class ExposureCapTests(unittest.TestCase):
    def test_gross_exposure_headroom_clips_notional(self):
        verdict = assess_new_position(
            symbol="NVDA",
            requested_notional=30_000.0,
            equity=100_000.0,
            open_positions={"AAPL": 80_000.0},
            price_history={},
            config=_config(max_gross_exposure_pct=100.0),
        )
        # Headroom is 100k - 80k = 20k.
        self.assertEqual(verdict.adjusted_notional, 20_000.0)
        self.assertTrue(any("exposure" in r.lower() for r in verdict.reasons))

    def test_no_headroom_zeroes_the_trade_with_reason(self):
        verdict = assess_new_position(
            symbol="NVDA",
            requested_notional=10_000.0,
            equity=100_000.0,
            open_positions={"AAPL": 100_000.0},
            price_history={},
            config=_config(max_gross_exposure_pct=100.0),
        )
        self.assertEqual(verdict.adjusted_notional, 0.0)
        self.assertFalse(verdict.allowed)


class RobustnessTests(unittest.TestCase):
    def test_missing_price_data_keeps_full_size_with_warning(self):
        verdict = assess_new_position(
            symbol="NEW",
            requested_notional=10_000.0,
            equity=100_000.0,
            open_positions={"AAPL": 10_000.0},
            price_history={},  # nothing available
            config=_config(),
        )
        self.assertEqual(verdict.adjusted_notional, 10_000.0)
        self.assertTrue(verdict.allowed)
        self.assertTrue(any("unavailable" in r.lower() for r in verdict.reasons))

    def test_min_size_factor_floors_combined_penalties(self):
        closes = _trending()
        rng = np.random.default_rng(11)
        wild = list(
            np.array(closes) * np.cumprod(1 + rng.normal(0, 0.06, len(closes)))
        )
        config = _config(min_size_factor=0.25, target_daily_vol_pct=0.5)
        verdict = assess_new_position(
            symbol="MEME",
            requested_notional=10_000.0,
            equity=1_000_000.0,
            open_positions={"AAPL": 10_000.0},
            price_history={"MEME": _frame(wild), "AAPL": _frame(closes)},
            config=config,
        )
        self.assertGreaterEqual(verdict.adjusted_notional, 2_500.0)

    def test_adjust_helper_only_touches_new_long_exposure(self):
        # SELL/HOLD and closes pass through untouched even with data present.
        for action in ("SELL", "HOLD", "NEUTRAL"):
            amount = adjust_new_position_notional(
                symbol="AAPL",
                action=action,
                requested_notional=5_000.0,
                gather_state=lambda: (100_000.0, {"MSFT": 50_000.0}, {}),
                config=_config(),
            )
            self.assertEqual(amount, 5_000.0)

    def test_adjust_helper_survives_gather_failure(self):
        def broken_gather():
            raise ConnectionError("broker down")

        amount = adjust_new_position_notional(
            symbol="AAPL",
            action="BUY",
            requested_notional=5_000.0,
            gather_state=broken_gather,
            config=_config(),
        )
        self.assertEqual(amount, 0.0)


class ConfigTests(unittest.TestCase):
    def test_default_config_exposes_portfolio_keys(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        self.assertIn("portfolio_intelligence_enabled", DEFAULT_CONFIG)
        self.assertIn("portfolio_high_correlation", DEFAULT_CONFIG)
        self.assertIn("portfolio_max_gross_exposure_pct", DEFAULT_CONFIG)

    def test_from_config_reads_project_keys(self):
        config = PortfolioLimitsConfig.from_config(
            {
                "portfolio_high_correlation": 0.5,
                "portfolio_max_gross_exposure_pct": 80.0,
            }
        )
        self.assertEqual(config.high_correlation, 0.5)
        self.assertEqual(config.max_gross_exposure_pct, 80.0)


if __name__ == "__main__":
    unittest.main()
