from __future__ import annotations

import json

import pytest

from arena.llm_pricing import estimate_llm_cost, reset_pricing_catalog_cache


@pytest.fixture(autouse=True)
def _reset_pricing_catalog(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ARENA_LLM_PRICING_CATALOG_JSON", raising=False)
    monkeypatch.delenv("ARENA_LLM_PRICING_CATALOG_PATH", raising=False)
    reset_pricing_catalog_cache()
    yield
    reset_pricing_catalog_cache()


def test_openai_cost_uses_cached_input_discount_and_output_rate() -> None:
    cost = estimate_llm_cost(
        provider="openai",
        model="gpt-5.2",
        usage={
            "prompt_tokens": 10_000,
            "cached_tokens": 6_000,
            "completion_tokens": 1_000,
            "thinking_tokens": 500,
        },
    )

    assert cost["input_tokens"] == 10_000
    assert cost["cache_read_input_tokens"] == 6_000
    assert cost["cached_input_tokens"] == 6_000
    assert cost["uncached_input_tokens"] == 4_000
    assert cost["output_tokens"] == 1_500
    assert cost["raw_total_tokens"] == 11_500
    assert cost["input_cost_usd"] == pytest.approx(0.007)
    assert cost["cache_read_cost_usd"] == pytest.approx(0.00105)
    assert cost["cached_input_cost_usd"] == pytest.approx(0.00105)
    assert cost["output_cost_usd"] == pytest.approx(0.021)
    assert cost["estimated_cost_usd"] == pytest.approx(0.02905)
    assert cost["pricing_status"] == "estimated"


def test_anthropic_cost_separates_cache_read_and_cache_write_tokens() -> None:
    cost = estimate_llm_cost(
        provider="anthropic",
        model="claude-opus-4-1-20250805",
        usage={
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "input_tokens": 1_000,
            "cache_read_input_tokens": 500,
            "cache_creation_input_tokens": 200,
            "output_tokens": 100,
        },
    )

    assert cost["input_tokens"] == 1_700
    assert cost["uncached_input_tokens"] == 1_000
    assert cost["cache_read_input_tokens"] == 500
    assert cost["cache_write_input_tokens"] == 200
    assert cost["raw_total_tokens"] == 1_800
    assert cost["input_cost_usd"] == pytest.approx(0.015)
    assert cost["cache_read_cost_usd"] == pytest.approx(0.00075)
    assert cost["cache_write_cost_usd"] == pytest.approx(0.00375)
    assert cost["output_cost_usd"] == pytest.approx(0.0075)
    assert cost["estimated_cost_usd"] == pytest.approx(0.027)
    assert cost["pricing_status"] == "family"


def test_anthropic_opus_47_uses_current_exact_price_before_opus_family_fallback() -> None:
    cost = estimate_llm_cost(
        provider="anthropic",
        model="claude-opus-4-7",
        usage={
            "prompt_tokens": 1_000,
            "cached_tokens": 200,
            "output_tokens": 100,
        },
    )

    assert cost["pricing_model"] == "claude-opus-4.7"
    assert cost["input_cost_usd"] == pytest.approx(0.004)
    assert cost["cached_input_cost_usd"] == pytest.approx(0.0001)
    assert cost["output_cost_usd"] == pytest.approx(0.0025)
    assert cost["estimated_cost_usd"] == pytest.approx(0.0066)


def test_gemini_cost_counts_thinking_tokens_as_output() -> None:
    cost = estimate_llm_cost(
        provider="gemini",
        model="gemini-3-pro-preview",
        usage={
            "prompt_tokens": 10_000,
            "cached_tokens": 4_000,
            "completion_tokens": 1_000,
            "thinking_tokens": 500,
        },
    )

    assert cost["input_tokens"] == 10_000
    assert cost["uncached_input_tokens"] == 6_000
    assert cost["cache_read_input_tokens"] == 4_000
    assert cost["output_tokens"] == 1_500
    assert cost["estimated_cost_usd"] == pytest.approx(0.0308)
    assert cost["pricing_status"] == "family"


def test_cached_input_is_clamped_to_prompt_tokens_for_subset_style_usage() -> None:
    cost = estimate_llm_cost(
        provider="openai",
        model="gpt-5.2",
        usage={
            "prompt_tokens": 100,
            "cached_tokens": 500,
            "completion_tokens": 0,
        },
    )

    assert cost["input_tokens"] == 100
    assert cost["cache_read_input_tokens"] == 100
    assert cost["uncached_input_tokens"] == 0
    assert cost["raw_total_tokens"] == 100


def test_pricing_catalog_env_adds_new_model_without_code_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "ARENA_LLM_PRICING_CATALOG_JSON",
        json.dumps(
            {
                "rules": [
                    {
                        "provider": "openai",
                        "model_id": "new-frontier-model-2026",
                        "pricing_model": "new-frontier-model-2026",
                        "input_usd_per_mtok": 2.0,
                        "cache_read_usd_per_mtok": 0.2,
                        "output_usd_per_mtok": 20.0,
                        "pricing_status": "configured",
                    }
                ]
            }
        ),
    )
    reset_pricing_catalog_cache()

    cost = estimate_llm_cost(
        provider="openai",
        model="new-frontier-model-2026",
        usage={"prompt_tokens": 1_000, "cached_tokens": 500, "completion_tokens": 100},
    )

    assert cost["pricing_model"] == "new-frontier-model-2026"
    assert cost["pricing_status"] == "configured"
    assert cost["estimated_cost_usd"] == pytest.approx(0.0031)


def test_unknown_model_keeps_token_breakdown_without_inventing_cost() -> None:
    cost = estimate_llm_cost(
        provider="new-provider",
        model="future-model-x",
        usage={"prompt_tokens": 1_000, "cached_tokens": 200, "completion_tokens": 100},
    )

    assert cost["input_tokens"] == 1_000
    assert cost["cache_read_input_tokens"] == 200
    assert cost["output_tokens"] == 100
    assert cost["estimated_cost_usd"] == 0.0
    assert cost["pricing_status"] == "unknown"


def test_new_major_model_is_unknown_until_pricing_catalog_is_configured() -> None:
    cost = estimate_llm_cost(
        provider="openai",
        model="gpt-6",
        usage={"prompt_tokens": 1_000, "cached_tokens": 200, "completion_tokens": 100},
    )

    assert cost["provider"] == "openai"
    assert cost["pricing_model"] == "gpt-6"
    assert cost["input_tokens"] == 1_000
    assert cost["estimated_cost_usd"] == 0.0
    assert cost["pricing_status"] == "unknown"


def test_openai_recent_frontier_models_use_explicit_catalog_prices() -> None:
    cost = estimate_llm_cost(
        provider="openai",
        model="gpt-5.5",
        usage={"prompt_tokens": 1_000, "cached_tokens": 200, "completion_tokens": 100},
    )

    assert cost["pricing_model"] == "gpt-5.5"
    assert cost["pricing_status"] == "estimated"
    assert cost["estimated_cost_usd"] == pytest.approx(0.0071)


def test_openai_pro_model_without_cache_rate_charges_uncached_and_output() -> None:
    cost = estimate_llm_cost(
        provider="openai",
        model="gpt-5.4-pro",
        usage={"prompt_tokens": 1_000, "completion_tokens": 100},
    )

    assert cost["pricing_model"] == "gpt-5.4-pro"
    assert cost["input_cost_usd"] == pytest.approx(0.03)
    assert cost["output_cost_usd"] == pytest.approx(0.018)
    assert cost["estimated_cost_usd"] == pytest.approx(0.048)
