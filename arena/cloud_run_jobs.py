from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def run_cloud_run_job(
    *,
    project: str,
    region: str,
    job_name: str,
    body: dict[str, Any] | None = None,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Starts one Cloud Run Job execution via the REST API."""
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    clean_project = str(project or "").strip()
    clean_region = str(region or "").strip()
    clean_job = str(job_name or "").strip()
    if not clean_project:
        raise ValueError("project is required")
    if not clean_region:
        raise ValueError("region is required")
    if not clean_job:
        raise ValueError("job_name is required")

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(creds)
    url = (
        f"https://{clean_region}-run.googleapis.com/apis/run.googleapis.com/v1/"
        f"namespaces/{clean_project}/jobs/{clean_job}:run"
    )

    response = session.post(url, json=body or {}, timeout=max(1, int(timeout_seconds)))
    if response.status_code >= 400:
        raise RuntimeError(
            f"cloud run job dispatch failed: job={clean_job} status={response.status_code} body={response.text[:500]}"
        )
    try:
        return response.json()
    except ValueError:
        logger.info("[cyan]Cloud Run job dispatched[/cyan] job=%s status=%d", clean_job, response.status_code)
        return {}
