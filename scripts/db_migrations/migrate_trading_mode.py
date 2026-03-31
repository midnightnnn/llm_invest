from arena.config import load_settings
from arena.data.bq import BigQueryRepository

def migrate():
    settings = load_settings()
    repo = BigQueryRepository(settings.google_cloud_project, settings.bq_dataset, settings.bq_location)
    
    print("Migrating agent_memory_events...")
    try:
        repo.execute(f"ALTER TABLE `{repo.dataset_fqn}.agent_memory_events` ADD COLUMN trading_mode STRING;")
        print("Column trading_mode added to agent_memory_events.")
    except Exception as e:
        print("Could not add column to agent_memory_events:", e)
        
    try:
        repo.execute(f"""
        UPDATE `{repo.dataset_fqn}.agent_memory_events` 
        SET trading_mode = 'paper' 
        WHERE trading_mode IS NULL;
        """)
        print("Updated old memories to 'paper'.")
    except Exception as e:
        print("Update failed:", e)

    print("Migrating board_posts...")
    try:
        repo.execute(f"ALTER TABLE `{repo.dataset_fqn}.board_posts` ADD COLUMN trading_mode STRING;")
        print("Column trading_mode added to board_posts.")
    except Exception as e:
        print("Could not add column to board_posts:", e)
        
    try:
        repo.execute(f"""
        UPDATE `{repo.dataset_fqn}.board_posts` 
        SET trading_mode = 'paper' 
        WHERE trading_mode IS NULL;
        """)
        print("Updated old posts to 'paper'.")
    except Exception as e:
        print("Update failed:", e)

if __name__ == '__main__':
    migrate()
