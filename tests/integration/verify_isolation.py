"""Manual BigQuery isolation verification script. Not part of the pytest suite."""

from arena.config import load_settings
from arena.data.bq import BigQueryRepository
from arena.models import BoardPost

def verify():
    settings = load_settings()
    repo = BigQueryRepository(settings.google_cloud_project, settings.bq_dataset, settings.bq_location)

    print("--- BigQuery SQL 격리 테스트 ---")
    
    # Dummy 게시글 생성
    live_post = BoardPost(
        agent_id="test_agent", title="[LIVE_TEST] 실전 매매 기록", body="실제 체결된 기록", trading_mode="live", tickers=["TEST"]
    )
    paper_post = BoardPost(
        agent_id="test_agent", title="[PAPER_TEST] 모의 투자 기록", body="가상 체결 기록", trading_mode="paper", tickers=["TEST"]
    )
    
    repo.write_board_post(live_post)
    repo.write_board_post(paper_post)
    
    paper_posts = repo.recent_board_posts(limit=10, trading_mode="paper")
    live_posts = repo.recent_board_posts(limit=10, trading_mode="live")
    
    live_in_paper = any(p["title"] == "[LIVE_TEST] 실전 매매 기록" for p in paper_posts)
    paper_in_live = any(p["title"] == "[PAPER_TEST] 모의 투자 기록" for p in live_posts)
    
    print(f"Paper DB 조회 시 Live 데이터 혼입 없음: {not live_in_paper}")
    print(f"Live DB 조회 시 Paper 데이터 혼입 없음: {not paper_in_live}")

if __name__ == '__main__':
    verify()
