"""Exercise the real Alpaca SDK against a local HTTP broker boundary.

No credentials or network services are needed. The broker responses are
simulated; request serialization, SDK parsing and execution guards are real.
"""

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from alpaca.trading.client import TradingClient
import pytest

from tradingagents.dataflows.alpaca_utils import AlpacaUtils
from tradingagents.safety.guardrails import SafetyGuard


def order_payload(symbol="AAPL", side="buy", qty="9", status="accepted"):
    return {
        "id": str(uuid4()), "client_order_id": str(uuid4()),
        "created_at": "2026-01-05T15:00:00Z",
        "updated_at": "2026-01-05T15:00:00Z",
        "submitted_at": "2026-01-05T15:00:00Z",
        "symbol": symbol, "side": side, "qty": qty, "filled_qty": "0",
        "type": "market", "order_class": "simple", "time_in_force": "day",
        "status": status, "extended_hours": False,
    }


@pytest.fixture
def broker(monkeypatch, tmp_path):
    state = SimpleNamespace(
        requests=[], open_orders=[], reject_entries=False,
        fail_lookup=False, close_status="accepted", submissions=[],
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, body):
            content = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            state.requests.append(("GET", self.path))
            parsed = urlsplit(self.path)
            if parsed.path != "/v2/orders":
                return self.respond(404, {"message": "Unexpected route"})
            assert parse_qs(parsed.query)["status"] == ["open"]
            assert parse_qs(parsed.query)["symbols"] == ["AAPL"]
            if state.fail_lookup:
                return self.respond(422, {"message": "Lookup unavailable"})
            self.respond(200, state.open_orders)

        def do_POST(self):
            state.requests.append(("POST", self.path))
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.submissions.append(body)
            if state.reject_entries:
                return self.respond(422, {"code": 42210000, "message": "Order rejected"})
            result = order_payload(side=body["side"], qty=str(body["qty"]))
            result["order_class"] = body.get("order_class", "simple")
            result["time_in_force"] = body["time_in_force"]
            state.open_orders.append(result)
            self.respond(200, result)

        def do_DELETE(self):
            state.requests.append(("DELETE", self.path))
            self.respond(200, order_payload(side="sell", qty="5", status=state.close_status))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(
        "tradingagents.dataflows.alpaca_utils.get_alpaca_trading_client",
        lambda: TradingClient("sandbox-key", "sandbox-secret", paper=True, url_override=url),
    )
    monkeypatch.setattr(AlpacaUtils, "get_latest_quote", lambda _: {"bid_price": 100, "ask_price": 110})
    monkeypatch.setattr(
        AlpacaUtils, "_safety_context",
        lambda _: ({"equity": 100_000, "last_equity": 100_000}, 0),
    )
    guard = SafetyGuard(state_path=tmp_path / "state.json", kill_switch_path=tmp_path / "STOP")
    monkeypatch.setattr("tradingagents.safety.get_safety_guard", lambda: guard)
    monkeypatch.setattr("tradingagents.alerts.notify_safety_block", lambda *_: None)
    state.guard = guard
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def protected_buy():
    return AlpacaUtils.execute_trading_action(
        "AAPL", "NEUTRAL", "BUY", 1000,
        protective_prices={"stop_loss_price": 95, "take_profit_price": 120},
    )


def test_protected_entry_serializes_and_parses_real_sdk_order(broker):
    result = protected_buy()
    assert result["success"]
    assert [method for method, _ in broker.requests] == ["GET", "POST"]
    request = broker.submissions[0]
    assert request["symbol"] == "AAPL"
    assert request["qty"] == 9  # The ask, not the bid, controls affordability.
    assert request["order_class"] == "bracket"
    assert request["time_in_force"] == "gtc"
    assert request["stop_loss"] == {"stop_price": 95.0}
    assert request["take_profit"] == {"limit_price": 120.0}
    assert result["actions"][0]["result"]["order_id"]


def test_rejected_protection_never_retries_without_protection(broker):
    broker.reject_entries = True
    result = protected_buy()
    assert not result["success"]
    assert result["actions"][0]["result"]["protective_failed"]
    assert len(broker.submissions) == 1
    assert broker.submissions[0]["order_class"] == "bracket"


def test_concurrent_entries_submit_only_once_when_broker_reports_pending(broker):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: protected_buy(), range(2)))
    assert sum(result["success"] for result in results) == 1
    assert len(broker.submissions) == 1
    blocked = next(result for result in results if not result["success"])
    assert blocked["actions"][0]["result"]["broker_attempted"] is False


def test_failed_pending_order_lookup_blocks_submission(broker):
    broker.fail_lookup = True
    assert not protected_buy()["success"]
    assert not broker.submissions


def test_partial_close_sends_fifty_percent_on_the_wire(broker):
    result = AlpacaUtils.close_position("AAPL", 50)
    assert result["success"]
    method, path = broker.requests[0]
    assert method == "DELETE"
    assert urlsplit(path).path == "/v2/positions/AAPL"
    assert parse_qs(urlsplit(path).query) == {"percentage": ["50.0"]}


@pytest.mark.parametrize("status", ["accepted", "partially_filled"])
def test_pending_or_partial_exit_prevents_reversal_entry(broker, status):
    broker.close_status = status
    result = AlpacaUtils.execute_trading_action("AAPL", "LONG", "SHORT", 1000, allow_shorts=True)
    assert result["success"] and result["pending_close"]
    assert [method for method, _ in broker.requests] == ["DELETE"]
    assert not broker.submissions


def test_kill_switch_prevents_any_broker_request(broker):
    broker.guard.kill_switch_path.write_text("Integration test halt", encoding="utf-8")
    result = protected_buy()
    assert not result["success"] and result["safety_blocked"]
    assert not broker.requests
