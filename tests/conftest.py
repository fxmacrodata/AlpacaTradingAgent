"""Keep automated tests independent of credentials and personal safety state."""

from urllib.parse import urlsplit

import httpx
import pytest
import requests


@pytest.fixture(autouse=True)
def isolated_test_boundaries(monkeypatch, tmp_path):
    from tradingagents.safety import guardrails

    monkeypatch.setattr(guardrails, "_SAFETY_HOME", tmp_path / "safety")
    guardrails.reset_safety_guard()

    def require_local(url):
        host = urlsplit(str(url)).hostname
        if host not in {"localhost", "127.0.0.1", "::1"}:
            # pytest.fail escapes application error handlers, so an accidental
            # real API call cannot become a successful fallback in a unit test.
            pytest.fail(f"Unexpected external HTTP request to {host}; mock the service boundary")

    request = requests.sessions.Session.request
    send = httpx.Client.send
    async_send = httpx.AsyncClient.send

    def local_request(self, method, url, *args, **kwargs):
        require_local(url)
        return request(self, method, url, *args, **kwargs)

    def local_send(self, request, *args, **kwargs):
        require_local(request.url)
        return send(self, request, *args, **kwargs)

    async def local_async_send(self, request, *args, **kwargs):
        require_local(request.url)
        return await async_send(self, request, *args, **kwargs)

    monkeypatch.setattr(requests.sessions.Session, "request", local_request)
    monkeypatch.setattr(httpx.Client, "send", local_send)
    monkeypatch.setattr(httpx.AsyncClient, "send", local_async_send)
    yield
    guardrails.reset_safety_guard()
