"""미실현 손익(Unrealized PnL) 기반 과거 매수(BUY) 기억 점수(Mark-to-Market) 일괄 업데이트 스크립트.

보유 중인 포지션을 순회하며 현재 시장가를 기준으로 과거 BUY 이벤트의 Score를 매일 최신화한다.
"""
import argparse
import logging
import math

from google.cloud import firestore

from arena.config import load_settings
from arena.data.bq import BigQueryRepository
from arena.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def run_mtm_score_update(project: str, dataset: str, location: str, agent_ids: list[str], trading_mode: str, dry_run: bool = True) -> int:
    """모든 활성 에이전트의 현재 포지션을 기준으로 BUY 기억 점수를 MTM 업데이트한다."""
    repo = BigQueryRepository(project=project, dataset=dataset, location=location)
    db = firestore.Client(project=project)
    
    updated_count = 0
    
    for agent_id in agent_ids:
        snapshot, _, _ = repo.build_agent_sleeve_snapshot(
            agent_id=agent_id,
            include_simulated=(trading_mode != "live")
        )
        if not snapshot.positions:
            continue
            
        logger.info("[MTM] Agent %s: Checking %d open positions...", agent_id, len(snapshot.positions))
        
        for ticker, pos in snapshot.positions.items():
            market_price = pos.market_price_krw
            if market_price <= 0:
                continue
                
            buy_memories = repo.find_buy_memories_for_ticker(
                agent_id=agent_id,
                ticker=ticker,
                limit=10,  # 포지션이 유지되는 동안 관련된 10개의 최근 매수 기억을 추적
                trading_mode=trading_mode
            )
            
            for mem in buy_memories:
                buy_price = MemoryStore._extract_buy_price(mem)
                if buy_price <= 0:
                    continue
                    
                pnl_ratio = (market_price - buy_price) / buy_price
                new_score = max(0.1, min(0.5 + 0.5 * math.tanh(pnl_ratio * 3), 1.0))
                
                event_id = str(mem.get("event_id", "")).strip()
                old_score = float(mem.get("score") or 1.0)
                
                # 점수 변화가 거의 없으면 업데이트 생략 (최소 0.02 차이)
                if abs(new_score - old_score) < 0.02:
                    continue
                    
                if dry_run:
                    logger.info(
                        "[DRY RUN] Would update %s (%s) score: %.2f -> %.2f (Unrealized PnL: %+.2f%%)", 
                        event_id[:8], ticker, old_score, new_score, pnl_ratio * 100
                    )
                else:
                    repo.update_memory_score(event_id, new_score)
                    try:
                        doc_ref = db.collection("agent_memories").document(event_id)
                        doc_ref.update({"score": float(new_score)})
                    except Exception as fs_exc:
                        logger.warning("Firestore sync failed for %s: %s", event_id[:8], str(fs_exc))
                        
                    logger.info(
                        "[MTM Updated] %s (%s) score: %.2f -> %.2f (Unrealized PnL: %+.2f%%)", 
                        event_id[:8], ticker, old_score, new_score, pnl_ratio * 100
                    )
                updated_count += 1
                
    mode = "would be updated" if dry_run else "updated"
    logger.info("MTM Score Update complete: %d memory events %s", updated_count, mode)
    return updated_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="미실현 손익 기반 장기기억 점수 MTM 업데이트")
    parser.add_argument("--execute", action="store_true", help="실제 점수 업데이트 실행 (기본은 dry-run)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    
    settings = load_settings()
    run_mtm_score_update(
        project=settings.google_cloud_project,
        dataset=settings.bq_dataset,
        location=settings.bq_location,
        agent_ids=settings.agent_ids,
        trading_mode=settings.trading_mode,
        dry_run=not args.execute
    )
