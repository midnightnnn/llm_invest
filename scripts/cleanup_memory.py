"""Firestore 장기기억 정리 배치 스크립트.

180일 이상 경과 + score 0.3 미만인 agent_memories 문서를 삭제한다.
기본 dry-run 모드이며, --execute 플래그로 실제 삭제를 수행한다.
"""
import argparse
import logging
import os
from datetime import timedelta

from google.cloud import firestore

from arena.models import utc_now

logger = logging.getLogger(__name__)


def cleanup(project: str, max_age_days: int = 180, min_score: float = 0.3, dry_run: bool = True) -> int:
    """오래되고 낮은 점수의 기억 문서를 삭제한다."""
    db = firestore.Client(project=project)
    cutoff = utc_now() - timedelta(days=max_age_days)

    query = (
        db.collection("agent_memories")
        .where("created_at", "<", cutoff)
        .where("score", "<", min_score)
        .limit(500)
    )

    deleted = 0
    for doc in query.stream():
        if dry_run:
            logger.info("[DRY RUN] would delete %s (score=%.2f)", doc.id, doc.to_dict().get("score", 0))
        else:
            doc.reference.delete()
        deleted += 1

    mode = "would be deleted" if dry_run else "deleted"
    logger.info("Cleanup complete: %d documents %s", deleted, mode)
    return deleted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Firestore agent_memories 정리")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    parser.add_argument("--max-age-days", type=int, default=180)
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--execute", action="store_true", help="실제 삭제 실행 (기본은 dry-run)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cleanup(args.project, args.max_age_days, args.min_score, dry_run=not args.execute)
