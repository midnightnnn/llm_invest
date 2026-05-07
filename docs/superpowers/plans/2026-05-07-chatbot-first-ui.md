# 챗봇 우선 UI 재구성 — 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the investment chatbot to the landing page, compress the sidebar to 4 items, move the provider/model selector into settings, and route showcase mode through a read-only chatbot whose write tools are stripped at agent construction.

**Architecture:** A `read_only: bool` flag is threaded from the ADK app builder down through `build_investment_chat_agent` → `build_chat_registry` → `build_chat_tool_entries`. When true, `build_order_tool_entries` and `build_config_tool_entries` are skipped, so the agent simply does not have write tools. A second ADK mount at `/investment-chat/adk-readonly` builds an agent with that flag. The UI changes are mechanical: redirects, nav reduction, template trimming, and one new POST endpoint.

**Tech Stack:** FastAPI + Jinja2 templates (`arena/ui/`), Google ADK (`google.adk.cli.fast_api`), pytest (`tests/ui/`).

**Reference:** Spec at `docs/superpowers/specs/2026-05-07-chatbot-first-ui-design.md`.

---

## File Structure

**Files to create:**
- `arena/ui/templates/investment_chat_showcase_body.jinja2` — read-only chat page body (iframe only, no toast panels)
- `tests/ui/test_chat_read_only.py` — unit tests for the `read_only` flag through the agent builder chain
- `tests/ui/test_chatbot_first_ui.py` — route tests for `/`, settings banner, settings chat model card, `POST /settings/chat-model`, showcase redirect/page

**Files to modify:**
- `arena/prompts/prompt_pack.py` — add `read_only` to `render_investment_chat_instruction`
- `arena/agents/investment_chat/tools.py` — add `read_only` to `build_chat_tool_entries`
- `arena/agents/investment_chat/registry.py` — add `read_only` to `build_chat_registry`
- `arena/agents/investment_chat/factory.py` — add `read_only` to `build_investment_chat_agent`, propagate
- `arena/ui/investment_chat_adk.py` — add `read_only` to `InvestmentChatAgentLoader` and `build_investment_chat_adk_app`
- `arena/ui/app.py` — change root redirect, add second ADK mount
- `arena/ui/layout.py` — owner nav 4 items, showcase nav with chat at top
- `arena/ui/templates/investment_chat_body.jinja2` — remove provider/model selector form
- `arena/ui/routes/investment_chat.py` — keep iframe URL building, drop server-side rendering of selector form options
- `arena/ui/templates/settings_body.jinja2` — add chatbot banner
- `arena/ui/templates/settings_agents_panel.jinja2` — add Chat Provider/Model card
- `arena/ui/routes/settings_render_agents.py` — pass provider/model options + presets to template
- `arena/ui/routes/settings_admin.py` — add `POST /settings/chat-model` handler
- `arena/ui/routes/showcase.py` — change entry redirect, add chat route

---

## Task 1: Add `read_only` parameter to `PromptPack.render_investment_chat_instruction`

**Why first:** Lowest layer. Pure function, easy to test in isolation. Other tasks build on this.

**Files:**
- Modify: `arena/prompts/prompt_pack.py:97-113`
- Test: `tests/ui/test_chat_read_only.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/ui/test_chat_read_only.py
from __future__ import annotations

from arena.prompts.prompt_pack import PromptPack


def test_render_investment_chat_instruction_default_no_read_only_notice() -> None:
    text = PromptPack.render_investment_chat_instruction(
        tenant_id="local",
        provider="gpt",
        model_id="gpt-5.5",
    )
    assert "보기 전용" not in text


def test_render_investment_chat_instruction_read_only_appends_notice() -> None:
    text = PromptPack.render_investment_chat_instruction(
        tenant_id="local",
        provider="gpt",
        model_id="gpt-5.5",
        read_only=True,
    )
    assert "보기 전용" in text
    assert "주문" in text and "설정" in text  # mentions both blocked tool families
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
pytest tests/ui/test_chat_read_only.py -v
```

Expected: 2 tests, second fails with `TypeError: render_investment_chat_instruction() got an unexpected keyword argument 'read_only'`.

- [ ] **Step 3: Implement minimal change in `prompt_pack.py`**

Replace lines 96–113 of `arena/prompts/prompt_pack.py`:

```python
    @staticmethod
    def render_investment_chat_instruction(
        *,
        tenant_id: str,
        provider: str = "",
        model_id: str = "",
        read_only: bool = False,
    ) -> str:
        _ = provider, model_id
        tenant = str(tenant_id or "").strip().lower() or "local"
        text = render_prompt_text(
            "investment_chat",
            "system_prompt.txt",
            values={
                "tenant_id": tenant,
                "provider": provider,
                "model_id": model_id,
            },
        )
        if read_only:
            text = (
                text
                + "\n\n[showcase 보기 전용 세션] "
                "주문 제출과 설정 변경 도구는 이 세션에서 사용할 수 없습니다. "
                "조회/분석 도구만 사용해 답하세요."
            )
        return text
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
pytest tests/ui/test_chat_read_only.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add arena/prompts/prompt_pack.py tests/ui/test_chat_read_only.py
git commit -m "feat(prompt): add read_only flag to investment chat instruction"
```

---

## Task 2: Add `read_only` to `build_chat_tool_entries`

**Why:** This is where order/config tools get included. We skip them when `read_only=True`. Tests can rely on the fact that the function returns ToolEntry list and we can count by category/name.

**Files:**
- Modify: `arena/agents/investment_chat/tools.py:13-30`
- Test: `tests/ui/test_chat_read_only.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/ui/test_chat_read_only.py`:

```python
def test_build_chat_tool_entries_default_includes_order_and_config(monkeypatch) -> None:
    from arena.agents.investment_chat import tools as chat_tools
    from arena.config import load_settings
    from tests.ui.helpers import _DummyRepo

    settings = load_settings()
    repo = _DummyRepo()
    entries = chat_tools.build_chat_tool_entries(
        repo=repo,
        settings=settings,
        tenant_id="local",
    )
    names = {str(e.name or e.tool_id or "") for e in entries}
    assert "submit_approved_order" in names or any("submit" in n for n in names)
    assert any("config" in n.lower() or "approve_config" in n.lower() for n in names)


def test_build_chat_tool_entries_read_only_strips_order_and_config() -> None:
    from arena.agents.investment_chat import tools as chat_tools
    from arena.config import load_settings
    from tests.ui.helpers import _DummyRepo

    settings = load_settings()
    repo = _DummyRepo()
    entries = chat_tools.build_chat_tool_entries(
        repo=repo,
        settings=settings,
        tenant_id="local",
        read_only=True,
    )
    names = {str(e.name or e.tool_id or "") for e in entries}
    # No order submission, no config approval/apply tools.
    assert "submit_approved_order" not in names
    assert not any("approve_config" in n.lower() for n in names)
    assert not any("apply_config" in n.lower() for n in names)
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
pytest tests/ui/test_chat_read_only.py::test_build_chat_tool_entries_read_only_strips_order_and_config -v
```

Expected: FAIL with `TypeError: build_chat_tool_entries() got an unexpected keyword argument 'read_only'`.

- [ ] **Step 3: Implement the change in `arena/agents/investment_chat/tools.py`**

Replace the file contents:

```python
from __future__ import annotations

from typing import Any, Callable

from arena.agents.investment_chat.account_tools import build_account_tool_entries
from arena.agents.investment_chat.config_tools import build_config_tool_entries
from arena.agents.investment_chat.history_tools import build_history_tool_entries
from arena.agents.investment_chat.order_tools import build_order_tool_entries
from arena.config import Settings
from arena.tools.registry import ToolEntry


def build_chat_tool_entries(
    *,
    repo: Any,
    settings: Settings,
    tenant_id: str,
    invalidate_tenant_cache: Callable[..., Any] | None = None,
    read_only: bool = False,
) -> list[ToolEntry]:
    entries: list[ToolEntry] = [
        *build_account_tool_entries(repo=repo, settings=settings, tenant_id=tenant_id),
        *build_history_tool_entries(repo=repo, tenant_id=tenant_id),
    ]
    if not read_only:
        entries.extend(
            build_order_tool_entries(repo=repo, settings=settings, tenant_id=tenant_id)
        )
        entries.extend(
            build_config_tool_entries(
                repo=repo,
                settings=settings,
                tenant_id=tenant_id,
                invalidate_tenant_cache=invalidate_tenant_cache,
            )
        )
    return entries
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/ui/test_chat_read_only.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add arena/agents/investment_chat/tools.py tests/ui/test_chat_read_only.py
git commit -m "feat(chat): add read_only flag to build_chat_tool_entries"
```

---

## Task 3: Propagate `read_only` through `build_chat_registry`

**Files:**
- Modify: `arena/agents/investment_chat/registry.py:17-62`
- Test: `tests/ui/test_chat_read_only.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/ui/test_chat_read_only.py`:

```python
def test_build_chat_registry_read_only_strips_order_and_config() -> None:
    from arena.agents.investment_chat.registry import build_chat_registry
    from arena.config import load_settings
    from tests.ui.helpers import _DummyRepo

    settings = load_settings()
    repo = _DummyRepo()
    registry = build_chat_registry(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=None,
        read_only=True,
    )
    tool_ids = {str(entry.tool_id or "").lower() for entry in registry.list_entries()}
    assert "submit_approved_order" not in tool_ids
    assert not any("approve_config" in tid for tid in tool_ids)
    assert not any("apply_config" in tid for tid in tool_ids)
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
pytest tests/ui/test_chat_read_only.py::test_build_chat_registry_read_only_strips_order_and_config -v
```

Expected: FAIL with `TypeError: build_chat_registry() got an unexpected keyword argument 'read_only'`.

- [ ] **Step 3: Implement the change in `arena/agents/investment_chat/registry.py`**

Replace lines 17–62 of `arena/agents/investment_chat/registry.py`:

```python
def build_chat_registry(
    *,
    repo: Any,
    settings: Settings,
    tenant_id: str,
    registry: ToolRegistry | None,
    invalidate_tenant_cache: Callable[..., Any] | None = None,
    read_only: bool = False,
) -> ToolRegistry:
    tenant = normalize_tenant(tenant_id)
    tool_settings = account_scope_settings(repo, tenant_id=tenant, settings=settings)
    source = registry.clone() if registry is not None else build_default_registry(repo, tool_settings, tenant_id=tenant)
    disabled = _load_disabled_tool_ids(repo, tenant)
    chat_config = load_chat_agent_config(repo, tenant_id=tenant)
    if isinstance(chat_config.get("disabled_tools"), list):
        disabled.update(str(tool_id).strip() for tool_id in chat_config["disabled_tools"] if str(tool_id).strip())

    context_tools = _ContextTools(repo=repo, settings=tool_settings, agent_id=AGENT_ID, tenant_id=tenant)
    for tool_id, fn in {
        "search_past_experiences": context_tools.search_past_experiences,
        "search_peer_lessons": context_tools.search_peer_lessons,
        "get_research_briefing": context_tools.get_research_briefing,
        "portfolio_diagnosis": context_tools.portfolio_diagnosis,
        "trade_performance": context_tools.trade_performance,
    }.items():
        if source.get(tool_id) is not None:
            source.bind(tool_id, fn)

    entries: list[ToolEntry] = []
    for entry in source.list_entries(require_callable=True):
        token = str(entry.tool_id or entry.name or "").strip().lower()
        if token not in CHAT_ANALYSIS_TOOL_IDS:
            continue
        if any(marker in token for marker in WRITE_TOOL_MARKERS):
            continue
        entries.append(entry)
    entries.extend(
        build_chat_tool_entries(
            repo=repo,
            settings=tool_settings,
            tenant_id=tenant,
            invalidate_tenant_cache=invalidate_tenant_cache,
            read_only=read_only,
        )
    )
    return ToolRegistry(
        [entry for entry in entries if str(entry.tool_id or "").strip().lower() not in disabled]
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/ui/test_chat_read_only.py -v
pytest tests/ui/test_investment_chat_factory_core.py tests/ui/test_investment_chat_order_drafts.py -v
```

Expected: all pass. (Existing tests use `build_chat_registry` without `read_only`, and the default `read_only=False` keeps current behavior.)

- [ ] **Step 5: Commit**

```bash
git add arena/agents/investment_chat/registry.py tests/ui/test_chat_read_only.py
git commit -m "feat(chat): propagate read_only flag through build_chat_registry"
```

---

## Task 4: Propagate `read_only` through `build_investment_chat_agent`

**Files:**
- Modify: `arena/agents/investment_chat/factory.py:88-138`
- Test: `tests/ui/test_chat_read_only.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/ui/test_chat_read_only.py`:

```python
def test_build_investment_chat_agent_read_only_excludes_write_tools(monkeypatch) -> None:
    from types import SimpleNamespace

    from arena.agents.investment_chat import factory
    from arena.config import load_settings
    from tests.ui.helpers import _DummyRepo

    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    settings = load_settings()
    repo = _DummyRepo()
    agent = factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=None,
        read_only=True,
    )
    tool_names = {getattr(tool, "__name__", "") for tool in agent.tools}
    assert "submit_approved_order" not in tool_names
    assert not any("approve_config" in n.lower() for n in tool_names)
    assert not any("apply_config" in n.lower() for n in tool_names)
    assert "보기 전용" in agent.instruction
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
pytest tests/ui/test_chat_read_only.py::test_build_investment_chat_agent_read_only_excludes_write_tools -v
```

Expected: FAIL with `TypeError: build_investment_chat_agent() got an unexpected keyword argument 'read_only'`.

- [ ] **Step 3: Implement the change in `arena/agents/investment_chat/factory.py`**

Replace lines 88–138:

```python
def build_investment_chat_agent(
    *,
    repo: Any,
    settings: Settings,
    tenant_id: str,
    registry: ToolRegistry | None = None,
    provider: str | None = None,
    model_override: str | None = None,
    invalidate_tenant_cache: Callable[..., Any] | None = None,
    read_only: bool = False,
) -> Agent:
    tenant = normalize_tenant(tenant_id)
    chat_config = load_chat_agent_config(repo, tenant_id=tenant)
    tenant_provider, tenant_model = tenant_default_chat_selection(settings)
    provider_token = str(
        provider
        or chat_config.get("provider")
        or os.getenv("ARENA_CHAT_PROVIDER")
        or tenant_provider
        or "gemini"
    ).strip().lower() or "gemini"
    model_id = str(model_override or chat_config.get("model") or os.getenv("ARENA_CHAT_MODEL") or "").strip()
    if not model_id and provider_token == tenant_provider:
        model_id = tenant_model
    model_id = normalize_chat_model_selection(provider_token, model_id)
    llm_params = chat_config.get("llm_params") if isinstance(chat_config.get("llm_params"), dict) else {}
    max_tool_events = resolve_max_tool_events(settings)
    chat_registry = build_chat_registry(
        repo=repo,
        settings=settings,
        tenant_id=tenant,
        registry=registry,
        invalidate_tenant_cache=invalidate_tenant_cache,
        read_only=read_only,
    )
    tools = _wrapped_tools(chat_registry, repo=repo, settings=settings, tenant_id=tenant)
    model = _resolve_model(provider_token, settings, model_override=model_id, llm_params=llm_params)
    return Agent(
        name=APP_NAME,
        description="Arena 투자챗봇",
        model=model,
        instruction=PromptPack.render_investment_chat_instruction(
            tenant_id=tenant,
            provider=provider_token,
            model_id=model_id,
            read_only=read_only,
        ),
        tools=tools,
        generate_content_config=_build_generate_content_config(
            provider=provider_token,
            llm_params=llm_params,
            max_tool_events=max_tool_events,
        ),
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/ui/test_chat_read_only.py tests/ui/test_investment_chat_factory_core.py tests/ui/test_investment_chat_loader.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add arena/agents/investment_chat/factory.py tests/ui/test_chat_read_only.py
git commit -m "feat(chat): propagate read_only flag through build_investment_chat_agent"
```

---

## Task 5: Add `read_only` to `InvestmentChatAgentLoader` and `build_investment_chat_adk_app`

**Why:** This is where the agent loader passes the flag to `build_investment_chat_agent`, and where the FastAPI sub-app exposes the choice. The cache key must include `read_only` so owner and showcase agents do not collide.

**Files:**
- Modify: `arena/ui/investment_chat_adk.py:221-323` (loader class) and `arena/ui/investment_chat_adk.py:818-853` (builder)
- Test: `tests/ui/test_chat_read_only.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/ui/test_chat_read_only.py`:

```python
def test_loader_passes_read_only_to_factory(monkeypatch) -> None:
    from arena.ui import investment_chat_adk

    captured: dict[str, object] = {}

    def fake_build_agent(**kwargs):
        captured.update(kwargs)
        from types import SimpleNamespace
        return SimpleNamespace(name="fake", instruction="x", tools=[])

    monkeypatch.setattr(investment_chat_adk, "build_investment_chat_agent", fake_build_agent)

    from arena.config import load_settings
    from tests.ui.helpers import _DummyRepo

    settings = load_settings()

    loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=_DummyRepo(),
        settings_for_tenant=lambda _t: settings,
        get_default_registry=lambda _t: None,
        default_tenant="local",
        read_only=True,
    )
    monkeypatch.setattr(loader, "_selection", lambda *a, **k: ("gpt", "gpt-5.5"))
    loader.load_agent(investment_chat_adk.APP_NAME)
    assert captured.get("read_only") is True


def test_build_investment_chat_adk_app_accepts_read_only() -> None:
    from arena.ui import investment_chat_adk
    from arena.config import load_settings
    from tests.ui.helpers import _DummyRepo

    settings = load_settings()
    app = investment_chat_adk.build_investment_chat_adk_app(
        repo=_DummyRepo(),
        settings_for_tenant=lambda _t: settings,
        get_default_registry=lambda _t: None,
        default_tenant="local",
        url_prefix="/investment-chat/adk-readonly",
        read_only=True,
    )
    assert app is not None
```

- [ ] **Step 2: Run tests and confirm they fail**

```bash
pytest tests/ui/test_chat_read_only.py::test_loader_passes_read_only_to_factory tests/ui/test_chat_read_only.py::test_build_investment_chat_adk_app_accepts_read_only -v
```

Expected: FAIL with `TypeError: ... unexpected keyword argument 'read_only'`.

- [ ] **Step 3: Implement the changes**

In `arena/ui/investment_chat_adk.py`, modify the `InvestmentChatAgentLoader.__init__` (around line 221-260) to accept and store `read_only`:

Find this block (around line 221) and add the parameter:

```python
class InvestmentChatAgentLoader(BaseAgentLoader):
    def __init__(
        self,
        *,
        repo: Any,
        settings_for_tenant: Callable[[str], Settings],
        get_default_registry: Callable[[str], ToolRegistry],
        default_tenant: str,
        invalidate_tenant_cache: Callable[..., Any] | None = None,
        read_only: bool = False,
    ) -> None:
        self.repo = repo
        self.settings_for_tenant = settings_for_tenant
        self.get_default_registry = get_default_registry
        self.default_tenant = default_tenant
        self.invalidate_tenant_cache = invalidate_tenant_cache
        self.read_only = bool(read_only)
        self._cache: OrderedDict[str, Any] = OrderedDict()
```

(Keep whatever `__init__` body already exists — only add the `read_only` parameter and the `self.read_only = bool(read_only)` line. If your existing `__init__` differs structurally, adapt minimally.)

Modify `load_agent` (around line 284–323) to:

1. Include `self.read_only` in `cache_key`.
2. Pass `read_only=self.read_only` to `build_investment_chat_agent`.

Cache key change:

```python
cache_key = f"{agent_name}:{tenant}:{provider}:{model_id}:{int(self.read_only)}:{fingerprint}"
```

Builder call change:

```python
agent = build_investment_chat_agent(
    repo=self.repo,
    settings=settings,
    tenant_id=tenant,
    registry=None,
    provider=provider,
    model_override=model_id,
    invalidate_tenant_cache=self.invalidate_tenant_cache,
    read_only=self.read_only,
)
```

In `build_investment_chat_adk_app` (line 818–853), add the parameter and pass it to the loader:

```python
def build_investment_chat_adk_app(
    *,
    repo: Any,
    settings_for_tenant: Callable[[str], Settings],
    get_default_registry: Callable[[str], ToolRegistry],
    default_tenant: str,
    url_prefix: str = "/investment-chat/adk",
    auth_enabled: bool = False,
    current_user: CurrentUserFn | None = None,
    invalidate_tenant_cache: Callable[..., Any] | None = None,
    read_only: bool = False,
) -> FastAPI:
    loader = InvestmentChatAgentLoader(
        repo=repo,
        settings_for_tenant=settings_for_tenant,
        get_default_registry=get_default_registry,
        default_tenant=default_tenant,
        invalidate_tenant_cache=invalidate_tenant_cache,
        read_only=read_only,
    )
    # ... rest unchanged
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/ui/test_chat_read_only.py tests/ui/test_investment_chat_loader.py tests/ui/test_investment_chat_adk_auth.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add arena/ui/investment_chat_adk.py tests/ui/test_chat_read_only.py
git commit -m "feat(chat-adk): wire read_only flag into loader and ADK app builder"
```

---

## Task 6: Change root redirect from `/board` to `/investment-chat`

**Files:**
- Modify: `arena/ui/app.py:501-504`
- Test: `tests/ui/test_chatbot_first_ui.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/ui/test_chatbot_first_ui.py
from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_redirects_to_investment_chat(local_app) -> None:
    """Use existing fixture that builds the FastAPI app for tests."""
    client = TestClient(local_app, follow_redirects=False)
    resp = client.get("/")
    assert resp.status_code == 302
    assert resp.headers["location"].startswith("/investment-chat")


def test_root_redirects_preserve_tenant_query(local_app) -> None:
    client = TestClient(local_app, follow_redirects=False)
    resp = client.get("/?tenant_id=local")
    assert resp.status_code == 302
    assert "/investment-chat" in resp.headers["location"]
    assert "tenant_id=local" in resp.headers["location"]
```

If `local_app` fixture does not exist in `tests/ui/conftest.py`, search for the existing pattern other tests use (`grep -rn "TestClient" tests/ui/ | head -5`) and copy that. If a fixture is missing, add this minimal one to the new test file:

```python
import pytest

from arena.ui.app import build_app


@pytest.fixture
def local_app(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_DEFAULT_TENANT", "local")
    return build_app()
```

(Adjust the import if `build_app` lives at a different path — check `grep -n "def build_app\|FastAPI()" arena/ui/app.py`.)

- [ ] **Step 2: Run the test and confirm it fails**

```bash
pytest tests/ui/test_chatbot_first_ui.py::test_root_redirects_to_investment_chat -v
```

Expected: FAIL — current redirect goes to `/board`.

- [ ] **Step 3: Implement the change**

In `arena/ui/app.py`, replace lines 501–504:

```python
    @app.get("/")
    def _root_redirect(tenant_id: str = "") -> RedirectResponse:
        qs = f"?tenant_id={tenant_id}" if tenant_id else ""
        return RedirectResponse(url=f"/investment-chat{qs}", status_code=302)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/ui/test_chatbot_first_ui.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add arena/ui/app.py tests/ui/test_chatbot_first_ui.py
git commit -m "feat(ui): redirect / to /investment-chat as new landing"
```

---

## Task 7: Reduce owner sidebar nav to 4 items + add chat to showcase nav

**Files:**
- Modify: `arena/ui/layout.py:33-65`
- Test: `tests/ui/test_chatbot_first_ui.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/ui/test_chatbot_first_ui.py`:

```python
def test_owner_nav_has_four_items_with_chat_first() -> None:
    from arena.ui.layout import tailwind_layout

    html = tailwind_layout("Test", "<p>body</p>", active="board", tenant="local")
    # Owner nav: only 4 sidebar items.
    assert html.count('class="sidebar-link') >= 4  # 4 nav links + maybe logout
    assert "투자챗봇" in html
    assert "게시판" in html
    assert "운용성과" in html
    assert "환경설정" in html
    # Removed items.
    assert "에이전트" not in html or html.index("투자챗봇") < html.index("에이전트")  # not a sidebar link anymore
    assert "자본관리" not in html or "자본관리" not in html.split('<aside')[1].split('</aside>')[0]
    # Chat is first.
    chat_idx = html.index("투자챗봇")
    board_idx = html.index("게시판")
    assert chat_idx < board_idx


def test_showcase_nav_has_chat_first() -> None:
    from arena.ui.layout import tailwind_layout

    html = tailwind_layout(
        "Test", "<p>body</p>", active="board", tenant="acme", showcase=True,
    )
    aside = html.split('<aside')[1].split('</aside>')[0]
    assert "투자챗봇" in aside
    chat_idx = aside.index("투자챗봇")
    board_idx = aside.index("게시판")
    assert chat_idx < board_idx
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
pytest tests/ui/test_chatbot_first_ui.py::test_owner_nav_has_four_items_with_chat_first tests/ui/test_chatbot_first_ui.py::test_showcase_nav_has_chat_first -v
```

Expected: FAIL — current nav has 7 items and chat is last.

- [ ] **Step 3: Implement the change in `arena/ui/layout.py`**

Replace lines 33–55:

```python
    if showcase:
        _t = html.escape(tenant or "")
        nav_items: list[tuple[str, str, str]] = [
            (f"/showcase/{_t}/investment-chat", "투자챗봇", "investment_chat"),
            (f"/showcase/{_t}/board", "게시판", "board"),
            (f"/showcase/{_t}/nav", "운용성과", "nav"),
            (f"/showcase/{_t}/settings?tab=agents", "에이전트", "agents"),
            (f"/showcase/{_t}/settings?tab=capital", "자본관리", "capital"),
            (f"/showcase/{_t}/settings?tab=mcp", "도구관리", "tools"),
            (f"/showcase/{_t}/settings?tab=memory", "기억관리", "memory"),
        ]
    else:
        tenant_query = f"?tenant_id={quote(str(tenant).strip().lower())}" if str(tenant or "").strip() else ""
        nav_items: list[tuple[str, str, str]] = [
            (f"/investment-chat{tenant_query}", "투자챗봇", "investment_chat"),
            ("/board", "게시판", "board"),
            ("/nav", "운용성과", "nav"),
            ("/settings?tab=agents", "환경설정", "settings"),
        ]
```

(Note the owner nav now has a single `환경설정` entry pointing at `?tab=agents`. The four old entries — 에이전트/자본관리/도구관리/기억관리 — are gone from the sidebar; users still reach those tabs via the tab strip on the settings page.)

- [ ] **Step 4: Run tests**

```bash
pytest tests/ui/test_chatbot_first_ui.py tests/ui/test_investment_chat_layout.py -v
```

Expected: new tests pass; existing layout tests should still pass (they test cosmetic things not nav size — but if any reference the removed labels, update them).

If any existing test asserts on owner nav containing `에이전트` etc., update it to expect `환경설정` instead. Check:

```bash
grep -rnE 'sidebar-link|에이전트|자본관리|도구관리|기억관리' tests/ui/ | head -10
```

- [ ] **Step 5: Commit**

```bash
git add arena/ui/layout.py tests/ui/test_chatbot_first_ui.py
git commit -m "feat(ui): reduce owner sidebar to 4 items, put chat first in showcase nav"
```

---

## Task 8: Add `/investment-chat/adk-readonly` mount in `app.py`

**Why:** Showcase chat iframe needs a separate ADK app whose loader uses `read_only=True`.

**Files:**
- Modify: `arena/ui/app.py:511-525` (the existing ADK mount block)
- Test: `tests/ui/test_chatbot_first_ui.py` (extend)

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_adk_readonly_mount_exists(local_app) -> None:
    paths = {route.path for route in local_app.routes if hasattr(route, "path")}
    # FastAPI mount creates a Mount route; check there is at least one route under /investment-chat/adk-readonly.
    assert any(p == "/investment-chat/adk-readonly" or p.startswith("/investment-chat/adk-readonly") for p in paths)
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
pytest tests/ui/test_chatbot_first_ui.py::test_adk_readonly_mount_exists -v
```

Expected: FAIL.

- [ ] **Step 3: Implement the change in `arena/ui/app.py`**

After the existing `app.mount("/investment-chat/adk", ...)` call (around line 511), add a second mount:

```python
    app.mount(
        "/investment-chat/adk-readonly",
        build_investment_chat_adk_app(
            repo=repo,
            settings_for_tenant=_settings_for_tenant,
            get_default_registry=_get_default_registry,
            default_tenant=_default_investment_chat_tenant(),
            url_prefix="/investment-chat/adk-readonly",
            auth_enabled=False,
            current_user=None,
            invalidate_tenant_cache=_invalidate_tenant_cache,
            read_only=True,
        ),
    )
```

(Match the exact argument names used by the original mount — refer to the lines just above. The key differences are: `auth_enabled=False`, `current_user=None`, and `read_only=True`.)

- [ ] **Step 4: Run tests**

```bash
pytest tests/ui/test_chatbot_first_ui.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add arena/ui/app.py tests/ui/test_chatbot_first_ui.py
git commit -m "feat(ui): mount read-only ADK at /investment-chat/adk-readonly"
```

---

## Task 9: Add settings banner template snippet + render in `settings_body.jinja2`

**Files:**
- Modify: `arena/ui/templates/settings_body.jinja2`
- Test: `tests/ui/test_chatbot_first_ui.py` (extend)

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_settings_page_shows_chatbot_banner(local_app) -> None:
    client = TestClient(local_app, follow_redirects=False)
    resp = client.get("/settings?tab=agents")
    # Auth may redirect; check the rendered HTML if 200 directly, else follow.
    if resp.status_code == 200:
        body = resp.text
    else:
        client = TestClient(local_app, follow_redirects=True)
        body = client.get("/settings?tab=agents").text
    assert "투자챗봇으로 변경 가능합니다" in body


def test_settings_banner_appears_on_all_tabs(local_app) -> None:
    client = TestClient(local_app, follow_redirects=True)
    for tab in ("agents", "capital", "mcp", "memory"):
        resp = client.get(f"/settings?tab={tab}")
        assert "투자챗봇으로 변경 가능합니다" in resp.text, f"missing banner on tab={tab}"
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
pytest tests/ui/test_chatbot_first_ui.py::test_settings_page_shows_chatbot_banner -v
```

Expected: FAIL.

- [ ] **Step 3: Implement the change**

Read the current `arena/ui/templates/settings_body.jinja2`, find the place between the page header (if any) and the tab strip. Insert the banner block at the very top of the body (before the first existing content):

```jinja
<div class="mb-4 flex items-center gap-2 rounded-xl border border-blue-200 bg-blue-50/80 px-4 py-2.5 text-xs text-blue-900 backdrop-blur">
  <svg class="h-4 w-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/></svg>
  <span>투자챗봇으로 변경 가능합니다.</span>
</div>
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/ui/test_chatbot_first_ui.py -v
```

Expected: banner tests pass.

- [ ] **Step 5: Commit**

```bash
git add arena/ui/templates/settings_body.jinja2 tests/ui/test_chatbot_first_ui.py
git commit -m "feat(settings): add chatbot banner to settings page header"
```

---

## Task 10: Add Chat Provider/Model card to `settings_agents_panel.jinja2` + wire data

**Why:** Provider/model selection moves from the chat page to settings. This task adds the form (UI). Task 11 adds the POST handler to actually save it.

**Files:**
- Modify: `arena/ui/templates/settings_agents_panel.jinja2` (top of agents panel)
- Modify: `arena/ui/routes/settings_render_agents.py` (or wherever agents tab is rendered) — pass `chat_provider_options`, `chat_model_options`, `chat_model_presets`, `chat_provider_current`, `chat_model_current` into the template
- Test: `tests/ui/test_chatbot_first_ui.py` (extend)

- [ ] **Step 1: Find where agents tab is rendered**

```bash
grep -nE 'render_settings_agents|settings_agents_panel|render_ui_template.*settings_agents' arena/ui/routes/*.py | head -10
```

Note the function and file. Use that path in subsequent steps.

- [ ] **Step 2: Write the failing test**

Append:

```python
def test_settings_agents_tab_has_chat_provider_card(local_app) -> None:
    client = TestClient(local_app, follow_redirects=True)
    body = client.get("/settings?tab=agents").text
    assert "Chat Provider/Model" in body or "투자챗봇 모델" in body
    # Form posts to /settings/chat-model
    assert 'action="/settings/chat-model"' in body
    # Provider select present
    assert 'name="provider"' in body
    assert 'name="model"' in body
```

- [ ] **Step 3: Run and confirm it fails**

```bash
pytest tests/ui/test_chatbot_first_ui.py::test_settings_agents_tab_has_chat_provider_card -v
```

Expected: FAIL.

- [ ] **Step 4: Implement the template addition**

At the top of `arena/ui/templates/settings_agents_panel.jinja2`, add (before the existing agent cards loop):

```jinja
{% if chat_provider_options %}
<section class="mb-5 rounded-2xl border border-ink-200/60 bg-white/80 p-4 backdrop-blur">
  <header class="mb-3 flex items-baseline justify-between">
    <h3 class="font-display text-sm font-semibold text-ink-900">Chat Provider/Model</h3>
    <p class="text-[11px] text-ink-500">투자챗봇이 사용하는 LLM. 페이지 진입 시 이 값이 적용됩니다.</p>
  </header>
  <form method="post" action="/settings/chat-model" class="flex flex-wrap items-center gap-2">
    <input type="hidden" name="tenant_id" value="{{ tenant }}" />
    <select name="provider" class="h-9 rounded-md border border-ink-200 bg-white px-2 text-xs font-semibold text-ink-700" data-chat-provider-settings>
      {% for item in chat_provider_options %}
      <option value="{{ item.value }}" {{ 'selected' if item.value == chat_provider_current else '' }}>{{ item.label }}</option>
      {% endfor %}
    </select>
    <select name="model" class="h-9 min-w-0 flex-1 rounded-md border border-ink-200 bg-white px-2 font-mono text-xs text-ink-700 sm:max-w-[460px]" data-chat-model-settings>
      {% for option in chat_model_options %}
      <option value="{{ option }}" {{ 'selected' if option == chat_model_current else '' }}>{{ option }}</option>
      {% endfor %}
    </select>
    <button type="submit" class="h-9 rounded-md bg-ink-900 px-3 text-xs font-semibold text-white hover:bg-ink-800">Apply</button>
  </form>
</section>
<script type="application/json" id="settings-chat-model-map">{{ chat_model_presets | tojson }}</script>
<script>
  (function() {
    var provider = document.querySelector('[data-chat-provider-settings]');
    var model = document.querySelector('[data-chat-model-settings]');
    var raw = document.getElementById('settings-chat-model-map');
    if (!provider || !model || !raw) return;
    var presets = {};
    try { presets = JSON.parse(raw.textContent || '{}') || {}; } catch (_e) { presets = {}; }
    provider.addEventListener('change', function() {
      var values = presets[provider.value] || [];
      model.innerHTML = '';
      values.forEach(function(v) {
        var opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        model.appendChild(opt);
      });
      if (values.length) model.value = values[0];
    });
  })();
</script>
{% endif %}
```

- [ ] **Step 5: Wire data into the agents tab renderer**

Open the file you found in Step 1 (likely `arena/ui/routes/settings_render_agents.py`). Locate the `render_ui_template("settings_agents_panel.jinja2", ...)` call. Add the new keyword arguments. Reuse the helpers from `arena/ui/routes/investment_chat.py` for now — copy them (we will dedupe later). At the top of the renderer file:

```python
from arena.ui.investment_chat_providers import tenant_available_provider_specs
from arena.providers.registry import canonical_provider, default_model_for_provider, list_adk_provider_specs
from arena.agents.investment_chat.selection import normalize_chat_model_selection, tenant_default_chat_selection
from arena.agents.investment_chat.config_tools import load_chat_agent_config


_CHAT_MODEL_PRESETS: dict[str, list[str]] = {
    "gemini": ["gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-2.5-flash", "gemini-2.5-pro"],
    "gpt": ["gpt-5.5", "gpt-5.2", "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"],
    "claude": ["claude-sonnet-4-6", "claude-opus-4-7", "claude-opus-4-5", "claude-sonnet-4-5"],
}


def _chat_model_options(provider: str, default_model: str, current_model: str = "") -> list[str]:
    provider_token = canonical_provider(provider) or str(provider or "").strip().lower()
    seen: set[str] = set()
    out: list[str] = []
    for token in [current_model, default_model, *_CHAT_MODEL_PRESETS.get(provider_token, [])]:
        value = normalize_chat_model_selection(provider_token, token)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _chat_provider_options(repo, tenant_id: str) -> list[dict[str, str]]:
    specs, _ = tenant_available_provider_specs(repo, tenant_id=tenant_id)
    return [{"value": s.provider_id, "label": s.label} for s in specs]


def _chat_model_preset_map(settings, provider_options: list[dict[str, str]]) -> dict[str, list[str]]:
    provider_ids = {item["value"] for item in provider_options}
    return {
        spec.provider_id: _chat_model_options(spec.provider_id, default_model_for_provider(settings, spec.provider_id))
        for spec in list_adk_provider_specs()
        if spec.provider_id in provider_ids
    }
```

In the function that calls `render_ui_template("settings_agents_panel.jinja2", …)`, before the call:

```python
chat_provider_options = _chat_provider_options(repo, tenant_id)
chat_config = load_chat_agent_config(repo, tenant_id=tenant_id)
tenant_default_provider, tenant_default_model = tenant_default_chat_selection(
    tenant_settings,
    allowed_providers={item["value"] for item in chat_provider_options},
)
chat_provider_current = (
    canonical_provider(chat_config.get("provider"))
    or tenant_default_provider
    or (chat_provider_options[0]["value"] if chat_provider_options else "")
)
chat_default_model = default_model_for_provider(tenant_settings, chat_provider_current) if chat_provider_current else ""
chat_model_current = normalize_chat_model_selection(
    chat_provider_current,
    chat_config.get("model") or chat_default_model,
)
chat_model_options = _chat_model_options(chat_provider_current, chat_default_model, chat_model_current)
chat_model_presets = _chat_model_preset_map(tenant_settings, chat_provider_options)
```

Then add to the `render_ui_template` call:

```python
chat_provider_options=chat_provider_options,
chat_model_options=chat_model_options,
chat_model_presets=chat_model_presets,
chat_provider_current=chat_provider_current,
chat_model_current=chat_model_current,
```

(Adapt variable names for `repo`, `tenant_id`, and `tenant_settings` to match the local context inside that function.)

- [ ] **Step 6: Run tests**

```bash
pytest tests/ui/test_chatbot_first_ui.py -v
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add arena/ui/templates/settings_agents_panel.jinja2 arena/ui/routes/settings_render_agents.py tests/ui/test_chatbot_first_ui.py
git commit -m "feat(settings): add chat provider/model card to agents tab"
```

---

## Task 11: Add `POST /settings/chat-model` handler

**Files:**
- Modify: `arena/ui/routes/settings_admin.py` (add route at end of `register_settings_admin_routes` or equivalent)
- Test: `tests/ui/test_chatbot_first_ui.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_chat_model_post_valid_saves_and_redirects(local_app, monkeypatch) -> None:
    client = TestClient(local_app, follow_redirects=False)
    # First read existing config (likely empty), then POST.
    resp = client.post(
        "/settings/chat-model",
        data={"tenant_id": "local", "provider": "gpt", "model": "gpt-5.5"},
    )
    assert resp.status_code in (302, 303)
    loc = resp.headers["location"]
    assert "/settings" in loc
    assert "tab=agents" in loc

    # Verify saved.
    from arena.agents.investment_chat.config_tools import load_chat_agent_config
    # Use the repo associated with the test app — fetch via dependency injection if available, or skip the storage check if not easy.


def test_chat_model_post_invalid_provider_rejects(local_app) -> None:
    client = TestClient(local_app, follow_redirects=False)
    resp = client.post(
        "/settings/chat-model",
        data={"tenant_id": "local", "provider": "no-such-provider", "model": "x"},
    )
    assert resp.status_code in (400, 422)
```

- [ ] **Step 2: Run and confirm it fails**

```bash
pytest tests/ui/test_chatbot_first_ui.py::test_chat_model_post_valid_saves_and_redirects -v
```

Expected: FAIL with 404 (route does not exist).

- [ ] **Step 3: Implement the route in `arena/ui/routes/settings_admin.py`**

Add inside `register_settings_admin_routes`:

```python
@app.post("/settings/chat-model")
def settings_chat_model(
    request: Request,
    tenant_id: str = Form(""),
    provider: str = Form(""),
    model: str = Form(""),
) -> RedirectResponse:
    from arena.agents.investment_chat.config_tools import load_chat_agent_config
    from arena.agents.investment_chat.selection import normalize_chat_model_selection
    from arena.providers.registry import canonical_provider
    from arena.ui.investment_chat_providers import tenant_available_provider_specs

    tenant, _agents, _user, redirect = deps.resolve_viewer_context(
        request,
        requested_tenant=tenant_id,
        next_path=f"/settings?tab=agents&tenant_id={tenant_id}",
    )
    if redirect is not None:
        return redirect

    provider_token = canonical_provider(provider) or str(provider or "").strip().lower()
    specs, _ = tenant_available_provider_specs(deps.repo, tenant_id=tenant)
    valid_provider_ids = {s.provider_id for s in specs}
    if provider_token not in valid_provider_ids:
        return JSONResponse({"error": "invalid provider"}, status_code=400)

    model_token = normalize_chat_model_selection(provider_token, model)
    if not model_token:
        return JSONResponse({"error": "invalid model"}, status_code=400)

    existing = load_chat_agent_config(deps.repo, tenant_id=tenant)
    merged = dict(existing)
    merged["provider"] = provider_token
    merged["model"] = model_token

    setter = getattr(deps.repo, "set_config", None)
    if not callable(setter):
        return JSONResponse({"error": "repo does not support set_config"}, status_code=500)
    try:
        setter(tenant, "investment_chat_config", json.dumps(merged, ensure_ascii=False))
    except TypeError:
        setter(tenant_id=tenant, config_key="investment_chat_config", value=json.dumps(merged, ensure_ascii=False))

    deps.invalidate_tenant_cache(tenant, "runtime", "memory", "portfolio")
    return RedirectResponse(url="/settings?tab=agents&saved=1", status_code=302)
```

Add necessary imports to the top of `arena/ui/routes/settings_admin.py` if missing:

```python
import json

from fastapi import Form
from fastapi.responses import JSONResponse, RedirectResponse
```

(Match the file's existing import organization.)

- [ ] **Step 4: Run tests**

```bash
pytest tests/ui/test_chatbot_first_ui.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add arena/ui/routes/settings_admin.py tests/ui/test_chatbot_first_ui.py
git commit -m "feat(settings): add POST /settings/chat-model handler"
```

---

## Task 12: Remove provider/model selector form from `investment_chat_body.jinja2`

**Why:** Now that the selector lives in settings, drop it from the chat page. The iframe and the two approval toast panels stay.

**Files:**
- Modify: `arena/ui/templates/investment_chat_body.jinja2` (remove lines 28–55 and the selector-related script block lines 126–154)
- Test: `tests/ui/test_chatbot_first_ui.py` (extend)

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_investment_chat_page_has_no_selector_form(local_app) -> None:
    client = TestClient(local_app, follow_redirects=True)
    resp = client.get("/investment-chat")
    body = resp.text
    # Selector form removed.
    assert "data-chat-selector-form" not in body
    assert "data-chat-provider" not in body
    assert "data-chat-model" not in body
    # iframe still present.
    assert "data-adk-chat-frame" in body
    # Approval toast panels still present.
    assert "data-order-draft-panel" in body
    assert "data-config-draft-panel" in body
```

- [ ] **Step 2: Run and confirm it fails**

```bash
pytest tests/ui/test_chatbot_first_ui.py::test_investment_chat_page_has_no_selector_form -v
```

Expected: FAIL — selector form is still in the template.

- [ ] **Step 3: Modify `arena/ui/templates/investment_chat_body.jinja2`**

Delete lines 28–55 (the `<form method="get" action="/investment-chat" ...>` block).

Delete lines 126–154 (the `<script>` block immediately after `</section>` that wires the model preset map and submit-on-change behavior — i.e. the script that references `data-chat-provider`, `data-chat-model`, `data-chat-selector-form`).

The first remaining content inside `<section class="investment-chat-shell ...">` should now begin with `<iframe ...>` directly.

Also remove the `{% if provider_options %}` / `{% else %}` / `{% endif %}` wrapper if it now wraps only the iframe + toasts — keep an empty-state fallback. Replace the structure with:

```jinja
<section class="investment-chat-shell flex flex-col overflow-hidden bg-white">
  {% if iframe_src %}
  <iframe
    src="{{ iframe_src }}"
    title="투자챗봇"
    class="investment-chat-frame min-h-0 flex-1 border-0"
    loading="eager"
    referrerpolicy="same-origin"
    data-adk-chat-frame
  ></iframe>
  <aside class="pointer-events-none fixed bottom-4 right-4 z-40 hidden w-[min(420px,calc(100vw-2rem))]" data-order-draft-panel>
    {# ...existing order draft panel markup unchanged... #}
  </aside>
  <aside class="pointer-events-none fixed bottom-4 left-4 z-40 hidden w-[min(440px,calc(100vw-2rem))]" data-config-draft-panel>
    {# ...existing config draft panel markup unchanged... #}
  </aside>
  {% else %}
  <div class="flex flex-1 items-center justify-center px-6">
    <div class="max-w-md rounded-lg border border-amber-200 bg-amber-50 p-5 text-sm leading-relaxed text-amber-900">
      등록된 LLM API key가 없거나 모델이 설정되지 않았습니다. 설정에서 챗봇 모델을 지정해 주세요.
    </div>
  </div>
  {% endif %}
</section>

{% if iframe_src %}
{# Approval polling scripts unchanged — keep them intact. #}
{# (existing order-draft script and config-draft script blocks at lines 156-432) #}
{% endif %}
```

(Keep both approval-polling `<script>` blocks intact. They reference `[data-order-draft-panel]` / `[data-config-draft-panel]` and post to `/investment-chat/order-drafts/...` etc. Wrap them in the same `{% if iframe_src %}` guard so they only render when the chat is actually loaded.)

- [ ] **Step 4: Update the route to drop now-unused template variables (optional, but tidy)**

Open `arena/ui/routes/investment_chat.py`. The `render_ui_template("investment_chat_body.jinja2", ...)` call passes `provider_options`, `model_options`, `model_presets`, `provider`, `model`, `tenant`. After removing the selector form, only `iframe_src` and `tenant` are still needed by the template. Trim:

```python
body = render_ui_template(
    "investment_chat_body.jinja2",
    iframe_src=_adk_iframe_src(tenant, provider_token, model_token) if provider_token and model_token else "",
    tenant=html.escape(tenant),
)
```

(The provider/model resolution code above stays — it builds the iframe URL.)

- [ ] **Step 5: Run tests**

```bash
pytest tests/ui/test_chatbot_first_ui.py tests/ui/test_investment_chat_pages.py -v
```

Expected: pass. If any existing test in `test_investment_chat_pages.py` asserts on the selector form, update it to assert it is gone (matching this spec change).

- [ ] **Step 6: Commit**

```bash
git add arena/ui/templates/investment_chat_body.jinja2 arena/ui/routes/investment_chat.py tests/ui/test_chatbot_first_ui.py
git commit -m "feat(chat): remove provider/model selector form from chat page"
```

---

## Task 13: Create `investment_chat_showcase_body.jinja2`

**Why:** showcase chat page needs only an iframe pointing to the read-only ADK mount. No toast panels, no polling.

**Files:**
- Create: `arena/ui/templates/investment_chat_showcase_body.jinja2`

- [ ] **Step 1: Create the template**

```jinja
<style>
  .investment-chat-shell {
    height: calc(100dvh - var(--mobile-topbar-h));
    min-height: 0;
  }
  .investment-chat-frame {
    display: block;
    width: 100%;
    min-width: 0;
    min-height: 0;
  }
  @media (min-width: 768px) {
    .investment-chat-shell {
      height: 100vh;
      min-height: 560px;
    }
  }
</style>
<section class="investment-chat-shell flex flex-col overflow-hidden bg-white">
  {% if iframe_src %}
  <iframe
    src="{{ iframe_src }}"
    title="투자챗봇 (보기 전용)"
    class="investment-chat-frame min-h-0 flex-1 border-0"
    loading="eager"
    referrerpolicy="same-origin"
    data-adk-chat-frame
  ></iframe>
  {% else %}
  <div class="flex flex-1 items-center justify-center px-6">
    <div class="max-w-md rounded-lg border border-amber-200 bg-amber-50 p-5 text-sm leading-relaxed text-amber-900">
      이 테넌트는 챗봇 모델이 설정되지 않아 미리보기를 표시할 수 없습니다.
    </div>
  </div>
  {% endif %}
</section>
```

- [ ] **Step 2: Commit (template alone — route uses it in next task)**

```bash
git add arena/ui/templates/investment_chat_showcase_body.jinja2
git commit -m "feat(showcase): add read-only chat body template"
```

---

## Task 14: Add `/showcase/{tenant}/investment-chat` route

**Files:**
- Modify: `arena/ui/routes/showcase.py`
- Test: `tests/ui/test_chatbot_first_ui.py` (extend)

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_showcase_investment_chat_page_renders(local_app) -> None:
    client = TestClient(local_app, follow_redirects=False)
    resp = client.get("/showcase/local/investment-chat")
    assert resp.status_code == 200
    body = resp.text
    # Iframe points to the read-only mount.
    assert "/investment-chat/adk-readonly" in body
    # No approval toast panels (write tools are absent so panels are pointless).
    assert "data-order-draft-panel" not in body
    assert "data-config-draft-panel" not in body
```

- [ ] **Step 2: Run and confirm it fails**

```bash
pytest tests/ui/test_chatbot_first_ui.py::test_showcase_investment_chat_page_renders -v
```

Expected: FAIL with 404.

- [ ] **Step 3: Implement the route in `arena/ui/routes/showcase.py`**

Add inside the route registration function (alongside the existing `/showcase/{tenant}/board` route):

```python
@app.get("/showcase/{tenant}/investment-chat", response_class=HTMLResponse)
def showcase_investment_chat(request: Request, tenant: str) -> HTMLResponse:
    from urllib.parse import urlencode
    from arena.providers.registry import canonical_provider, default_model_for_provider
    from arena.agents.investment_chat.selection import normalize_chat_model_selection, tenant_default_chat_selection
    from arena.agents.investment_chat.config_tools import load_chat_agent_config

    t = str(tenant or "").strip().lower()
    tenant_settings = deps.settings_for_tenant(t)
    chat_config = load_chat_agent_config(deps.repo, tenant_id=t)
    tenant_provider, tenant_model = tenant_default_chat_selection(tenant_settings)
    provider_token = (
        canonical_provider(chat_config.get("provider"))
        or tenant_provider
        or "gemini"
    )
    model_token = normalize_chat_model_selection(
        provider_token,
        chat_config.get("model")
        or default_model_for_provider(tenant_settings, provider_token)
        or tenant_model,
    )
    iframe_src = ""
    if provider_token and model_token:
        qs = urlencode({"tenant_id": t, "provider": provider_token, "model": model_token})
        iframe_src = f"/investment-chat/adk-readonly/dev-ui/?{qs}"
    body = render_ui_template(
        "investment_chat_showcase_body.jinja2",
        iframe_src=iframe_src,
    )
    return deps.html_response(
        deps.tailwind_layout(
            "투자챗봇",
            body,
            active="investment_chat",
            tenant=t,
            user=deps.current_user(request),
            max_width_class="max-w-none",
            hide_page_header=True,
            main_class="flex-1 min-w-0 w-full p-0 box-border",
            showcase=True,
        ),
        max_age=0,
    )
```

(Adapt to the surrounding code — match the existing showcase route pattern: imports, `deps` usage, `render_ui_template` import. Look at the existing `/showcase/{tenant}/board` route in the same file as a model.)

- [ ] **Step 4: Run tests**

```bash
pytest tests/ui/test_chatbot_first_ui.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add arena/ui/routes/showcase.py tests/ui/test_chatbot_first_ui.py
git commit -m "feat(showcase): add read-only investment chat page"
```

---

## Task 15: Change showcase entry redirect to `…/investment-chat`

**Files:**
- Modify: `arena/ui/routes/showcase.py:130-143` (the two redirect routes)
- Test: `tests/ui/test_chatbot_first_ui.py` (extend)

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_showcase_entry_redirects_to_investment_chat(local_app) -> None:
    client = TestClient(local_app, follow_redirects=False)
    resp = client.get("/showcase/local")
    assert resp.status_code == 302
    assert "/showcase/local/investment-chat" in resp.headers["location"]


def test_showcase_entry_with_trailing_slash(local_app) -> None:
    client = TestClient(local_app, follow_redirects=False)
    resp = client.get("/showcase/")
    # Existing handler redirects /showcase/ to a tenant-specific board; now should go to chat.
    # If your handler redirects to a default tenant, only assert path tail.
    assert resp.status_code == 302
    assert "investment-chat" in resp.headers["location"]
```

- [ ] **Step 2: Run and confirm it fails**

```bash
pytest tests/ui/test_chatbot_first_ui.py::test_showcase_entry_redirects_to_investment_chat -v
```

Expected: FAIL — redirect still goes to `…/board`.

- [ ] **Step 3: Implement the change in `arena/ui/routes/showcase.py`**

In both redirect blocks (lines ~130 and ~138 in the current file), change `f"/showcase/{tenant}/board"` to `f"/showcase/{tenant}/investment-chat"`:

```python
return HTMLResponse(status_code=302, headers={"Location": f"/showcase/{tenant}/investment-chat"})
```

```python
return HTMLResponse(status_code=302, headers={"Location": f"/showcase/{t}/investment-chat"})
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/ui/test_chatbot_first_ui.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add arena/ui/routes/showcase.py tests/ui/test_chatbot_first_ui.py
git commit -m "feat(showcase): redirect entry to investment-chat instead of board"
```

---

## Task 16: Final regression run + manual smoke test

**Why:** Catch any test that asserted on the old nav structure / selector form / showcase redirect.

**Files:** none (regression only)

- [ ] **Step 1: Run the entire UI test suite**

```bash
pytest tests/ui/ -v 2>&1 | tail -80
```

Expected: all pass. If any test fails because it asserted on removed elements (`/board` landing redirect, owner sidebar count == 7, selector form on chat page, etc.), update those tests to match the new behavior.

- [ ] **Step 2: Run the full test suite**

```bash
pytest -q 2>&1 | tail -20
```

Expected: green.

- [ ] **Step 3: Boot the UI locally and smoke test**

```bash
ARENA_DEFAULT_TENANT=local python -m arena ui --port 8080
```

Visit (in a browser):

1. `http://localhost:8080/` → should land on `/investment-chat` with 4-item sidebar.
2. Sidebar items: 투자챗봇 / 게시판 / 운용성과 / 환경설정.
3. `/settings?tab=agents` → blue "투자챗봇으로 변경 가능합니다" banner at top, "Chat Provider/Model" card, then existing agent cards.
4. Change provider/model in the card and click Apply → page reloads, dropdown shows the new selection.
5. Go back to `/investment-chat` → iframe URL should reflect the saved provider/model.
6. `http://localhost:8080/showcase/local` → redirects to `/showcase/local/investment-chat` with showcase sidebar (chat at top, no Logout button).
7. In showcase chat, ask: "주문해줘 — 테슬라 100주 매수." Agent should respond it cannot (no order tools).
8. In showcase chat, ask: "오늘 포트 알려줘." Agent should answer (read tools available).

- [ ] **Step 4: Commit any test fixes**

If you changed any tests in Step 1, commit them:

```bash
git add tests/...
git commit -m "test: update UI tests for chatbot-first navigation and showcase redirect"
```

---

## Self-review checklist (run before declaring done)

- [ ] Spec §1 Goals: every goal has a corresponding task. ✓ (Tasks 6/7/8 cover routing+nav; Tasks 9/10 cover settings; Tasks 1–5/8/13/14 cover showcase read-only.)
- [ ] Spec §3 Removed elements: provider/model selector form gone (Task 12). Page header still hidden via `hide_page_header=True`.
- [ ] Spec §4 Settings banner: Task 9. Chat Provider/Model card: Task 10. POST handler: Task 11.
- [ ] Spec §5 Showcase: redirect (Task 15), nav with chat first (Task 7), separate ADK mount (Task 8), template (Task 13), route (Task 14), agent-level read_only (Tasks 1–5).
- [ ] Spec §6 touch list: every file in the touch list is modified by some task.
- [ ] Placeholder scan: no "TBD", "TODO", "implement later" anywhere.
- [ ] Type/name consistency: `read_only` is the parameter name everywhere (not `readonly`, not `read-only`).
- [ ] Cache key in Task 5 includes `read_only` so owner/showcase agents do not collide.
