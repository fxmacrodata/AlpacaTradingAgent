"""
webui/components/cost_panel.py - LLM cost monitoring panel

Attributes LLM spend from the persisted run logs to analyses, days,
symbols, and models, next to each symbol's recorded outcome returns — so cost
without benefit is visible at a glance. Estimates are an upper bound
(cache-read discounts are not recorded) and unknown models are surfaced
as unpriced tokens rather than guessed.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html


def create_cost_panel():
    """Create the LLM cost panel card for the web UI."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4("LLM Cost Monitor", className="mb-1"),
                html.Div(
                    "Estimated spend from recorded token usage — attributed per "
                    "day, symbol, and model, alongside hypothetical outcome returns, "
                    "which are not realized account profits. Prices are "
                    "estimates (override via llm_pricing_per_million); cache "
                    "discounts are not tracked, so this is an upper bound.",
                    className="text-muted small mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                [html.I(className="fas fa-sync me-2"), "Refresh"],
                                id="cost-refresh-btn",
                                color="primary",
                                size="sm",
                            ),
                            width="auto",
                        ),
                    ],
                    className="g-2 mb-3",
                ),
                dcc.Loading(
                    id="cost-loading",
                    type="default",
                    children=html.Div(
                        [
                            html.Div(id="cost-summary-cards", className="mb-3"),
                            html.Div(
                                dcc.Graph(
                                    id="cost-daily-graph",
                                    config={"displayModeBar": False, "responsive": True},
                                    style={"height": "260px", "width": "100%"},
                                ),
                                id="cost-graph-container",
                                style={"display": "none"},
                            ),
                            html.Div(id="cost-symbol-table", className="mb-2"),
                            html.Div(id="cost-model-table"),
                        ]
                    ),
                ),
            ]
        ),
        className="mb-4",
    )
