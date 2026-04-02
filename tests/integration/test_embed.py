from __future__ import annotations

import os

import pytest

from tests.integration.conftest import require_live_integration

pytestmark = pytest.mark.integration


def test_vertex_embedding_smoke_returns_nonempty_vector() -> None:
    require_live_integration("GOOGLE_CLOUD_PROJECT")
    genai = pytest.importorskip("google.genai")
    location = str(os.getenv("GOOGLE_CLOUD_LOCATION") or "global").strip() or "global"
    if location == "global":
        location = "asia-northeast3"
    client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=location,
    )
    response = client.models.embed_content(
        model="text-embedding-004",
        contents="Apple earnings are out tomorrow.",
    )
    assert response.embeddings
    assert len(response.embeddings[0].values) > 0

