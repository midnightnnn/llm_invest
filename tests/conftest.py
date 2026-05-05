from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pytest

from tests.helpers.bigquery import FakeBigQuerySession

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tenant_id() -> str:
    return "tenant-a"


@pytest.fixture
def fixed_utc_now() -> datetime:
    return datetime(2026, 3, 29, 12, 30, tzinfo=timezone.utc)


@pytest.fixture
def fake_bq_session_factory() -> Callable[..., FakeBigQuerySession]:
    return FakeBigQuerySession
