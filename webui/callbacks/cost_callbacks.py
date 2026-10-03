"""
Cost callbacks for TradingAgents WebUI
Scans the persisted run logs, aggregates LLM spend per day/symbol/model,
joins each symbol's realized returns, and shows the daily token budget
state from the safety layer.
"""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, html

from webui.config.constants import COLORS


def _fmt_usd(value):
    return "—" if value is None else f"${value:,.2f}"


def _fmt_tokens(value):
    return f"{int(value or 0):,}"


def _summary_card(label, text, color=None):
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(label, className="text-muted small"),
                    html.Div(
                        text,
                        style={
                            "color": color or COLORS["text"],
                            "fontSize": "1.3rem",
                            "fontWeight": 600,
                        },
                    ),
                ],
                className="p-2 text-center",
            ),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}"},
        ),
        md=2,
        xs=6,
        className="mb-2",
    )


def _budget_text():
    """Daily token usage vs the safety layer's budget, if configured."""
    try:
        from tradingagents.safety import get_safety_guard

        guard = get_safety_guard()
        used = guard.llm_tokens_used()
        budget = float(guard.config.get("daily_llm_token_budget", 0) or 0)
        if budget > 0:
            return f"{used:,} / {budget:,.0f}", (
                COLORS["error"] if used >= budget else COLORS["completed"]
            )
        return f"{used:,} / ∞", COLORS["text"]
    except Exception:
        return "—", COLORS["pending"]


def _build_daily_figure(per_day):
    days = sorted(per_day)
    figure = go.Figure(
        go.Bar(
            x=days,
            y=[per_day[d]["cost_usd"] for d in days],
            marker_color=COLORS["primary"],
        )
    )
    figure.update_layout(
        title="Estimated LLM cost per day",
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        margin={"l": 40, "r": 20, "t": 40, "b": 30},
        yaxis={"title": "USD", "tickformat": ",.2f"},
        showlegend=False,
    )
    return figure


def _build_symbol_table(per_symbol, returns_by_symbol):
    if not per_symbol:
        return None
    header = html.Thead(
        html.Tr(
            [
                html.Th("Symbol"),
                html.Th("Runs"),
                html.Th("Tokens"),
                html.Th("Est. cost"),
                html.Th("Resolved decisions"),
                html.Th("Avg hypothetical return"),
            ]
        )
    )
    rows = []
    for symbol in sorted(per_symbol, key=lambda s: -per_symbol[s]["cost_usd"]):
        stats = per_symbol[symbol]
        outcome = returns_by_symbol.get(symbol) or {}
        avg = outcome.get("avg_return")
        avg_text = "—" if avg is None else f"{avg:+.2%}"
        avg_color = (
            COLORS["pending"]
            if avg is None
            else (COLORS["completed"] if avg > 0 else COLORS["error"])
        )
        rows.append(
            html.Tr(
                [
                    html.Td(symbol),
                    html.Td(stats["runs"]),
                    html.Td(_fmt_tokens(stats["total_tokens"])),
                    html.Td(_fmt_usd(stats["cost_usd"])),
                    html.Td(outcome.get("resolved", 0)),
                    html.Td(avg_text, style={"color": avg_color}),
                ]
            )
        )
    return html.Div(
        [
            html.H6("Per symbol — cost and hypothetical outcome", className="mt-2"),
            dbc.Table([header, html.Tbody(rows)], bordered=False, hover=True, size="sm", striped=True),
        ]
    )


def _build_model_table(per_model):
    if not per_model:
        return None
    header = html.Thead(
        html.Tr(
            [
                html.Th("Model"),
                html.Th("Input tokens"),
                html.Th("Output tokens"),
                html.Th("Est. cost"),
            ]
        )
    )
    rows = []
    for model in sorted(per_model, key=lambda m: -per_model[m]["cost_usd"]):
        stats = per_model[model]
        cost_text = _fmt_usd(stats["cost_usd"]) + (" (partly unpriced)" if stats.get("unpriced") else "")
        rows.append(
            html.Tr(
                [
                    html.Td(model),
                    html.Td(_fmt_tokens(stats["input_tokens"])),
                    html.Td(_fmt_tokens(stats["output_tokens"])),
                    html.Td(cost_text),
                ]
            )
        )
    return html.Div(
        [
            html.H6("Per model", className="mt-2"),
            dbc.Table([header, html.Tbody(rows)], bordered=False, hover=True, size="sm", striped=True),
        ]
    )


def register_cost_callbacks(app):
    """Register cost-panel callbacks with the Dash app"""

    @app.callback(
        [
            Output("cost-summary-cards", "children"),
            Output("cost-daily-graph", "figure"),
            Output("cost-graph-container", "style"),
            Output("cost-symbol-table", "children"),
            Output("cost-model-table", "children"),
        ],
        Input("cost-refresh-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_cost_panel(n_clicks):
        hidden = {"display": "none"}
        empty = go.Figure()
        try:
            from tradingagents.dataflows.config import get_config
            from tradingagents.llm_cost import (
                aggregate_costs,
                realized_returns_by_symbol,
                scan_run_costs,
            )

            config = {}
            try:
                config = dict(get_config() or {})
            except Exception:
                config = {}
            records = scan_run_costs(
                eval_results_dir=config.get("results_dir", "eval_results"),
                overrides=config.get("llm_pricing_per_million"),
            )
            aggregates = aggregate_costs(records)
            returns_by_symbol = realized_returns_by_symbol(config)
        except Exception as e:
            alert = dbc.Alert(f"Cost scan failed: {e}", color="danger", className="mb-0")
            return alert, empty, hidden, None, None

        totals = aggregates["totals"]
        if not totals["runs"]:
            alert = dbc.Alert(
                "No recorded runs found under eval_results/ yet.",
                color="info",
                className="mb-0",
            )
            return alert, empty, hidden, None, None

        budget_text, budget_color = _budget_text()
        cards = dbc.Row(
            [
                _summary_card("Analyses", f"{totals['runs']:,}"),
                _summary_card("Total tokens", _fmt_tokens(totals["total_tokens"])),
                _summary_card("Est. cost", _fmt_usd(totals["cost_usd"]), COLORS["primary"]),
                _summary_card("Unpriced tokens", _fmt_tokens(totals["unpriced_tokens"])),
                _summary_card("Today's tokens / budget", budget_text, budget_color),
            ],
            className="g-2",
        )
        figure = _build_daily_figure(aggregates["per_day"])
        symbol_table = _build_symbol_table(aggregates["per_symbol"], returns_by_symbol)
        model_table = _build_model_table(aggregates["per_model"])
        return cards, figure, {"display": "block"}, symbol_table, model_table
