from __future__ import annotations

import pytest


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = "fake response"

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")


class _FakeSession:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def get(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return _FakeResponse(self.payload)


class _PagedSession:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def get(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return _FakeResponse(self.payloads.pop(0))


def test_openai_model_discovery_filters_text_and_cheap_models() -> None:
    from arena.providers.model_discovery import discover_model_options_with_api_key

    session = _FakeSession(
        {
            "data": [
                {"id": "gpt-5.5"},
                {"id": "gpt-5.4-mini"},
                {"id": "gpt-image-1"},
                {"id": "text-embedding-3-large"},
                {"id": "whisper-1"},
            ]
        }
    )

    options = discover_model_options_with_api_key("gpt", "sk-test", session=session)

    assert options["advisor_models"] == ["gpt-5.5", "gpt-5.4-mini"]
    assert options["router_models"] == ["gpt-5.4-mini"]
    assert options["utility_models"] == ["gpt-5.4-mini"]
    assert session.calls[0]["headers"]["Authorization"] == "Bearer sk-test"


def test_gemini_model_discovery_requires_generate_content_and_strips_prefix() -> None:
    from arena.providers.model_discovery import discover_model_options_with_api_key

    session = _FakeSession(
        {
            "models": [
                {
                    "name": "models/gemini-2.5-pro",
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    "name": "models/gemini-2.5-flash",
                    "supportedGenerationMethods": ["generateContent", "countTokens"],
                },
                {
                    "name": "models/embedding-001",
                    "supportedGenerationMethods": ["embedContent"],
                },
            ]
        }
    )

    options = discover_model_options_with_api_key("gemini", "key-test", session=session)

    assert options["advisor_models"] == ["gemini-2.5-pro", "gemini-2.5-flash"]
    assert options["router_models"] == ["gemini-2.5-flash"]
    assert session.calls[0]["params"]["key"] == "key-test"


def test_claude_model_discovery_uses_haiku_for_cheap_models() -> None:
    from arena.providers.model_discovery import discover_model_options_with_api_key

    session = _FakeSession(
        {
            "data": [
                {"id": "claude-opus-4-1-20250805"},
                {"id": "claude-sonnet-4-5-20250929"},
                {"id": "claude-haiku-4-5-20251001"},
            ]
        }
    )

    options = discover_model_options_with_api_key("claude", "anthropic-key", session=session)

    assert options["advisor_models"] == [
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
    ]
    assert options["router_models"] == ["claude-haiku-4-5-20251001"]
    assert session.calls[0]["headers"]["x-api-key"] == "anthropic-key"


def test_gemini_model_discovery_follows_page_tokens() -> None:
    from arena.providers.model_discovery import discover_model_options_with_api_key

    session = _PagedSession(
        [
            {
                "models": [
                    {
                        "name": "models/gemini-2.5-pro",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ],
                "nextPageToken": "page-2",
            },
            {
                "models": [
                    {
                        "name": "models/gemini-2.5-flash",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ]
            },
        ]
    )

    options = discover_model_options_with_api_key("gemini", "key-test", session=session)

    assert options["advisor_models"] == ["gemini-2.5-pro", "gemini-2.5-flash"]
    assert session.calls[1]["params"]["pageToken"] == "page-2"


def test_claude_model_discovery_follows_after_id_pagination() -> None:
    from arena.providers.model_discovery import discover_model_options_with_api_key

    session = _PagedSession(
        [
            {
                "data": [{"id": "claude-sonnet-4-5-20250929"}],
                "has_more": True,
                "last_id": "claude-sonnet-4-5-20250929",
            },
            {"data": [{"id": "claude-haiku-4-5-20251001"}], "has_more": False},
        ]
    )

    options = discover_model_options_with_api_key("claude", "anthropic-key", session=session)

    assert options["advisor_models"] == ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"]
    assert session.calls[1]["params"]["after_id"] == "claude-sonnet-4-5-20250929"


def test_discovery_raises_when_no_text_models_are_available() -> None:
    from arena.providers.model_discovery import ModelDiscoveryError, discover_model_options_with_api_key

    session = _FakeSession({"data": [{"id": "text-embedding-3-large"}]})

    with pytest.raises(ModelDiscoveryError):
        discover_model_options_with_api_key("gpt", "sk-test", session=session)
