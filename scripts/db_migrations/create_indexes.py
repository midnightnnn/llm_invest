import os
import subprocess

def run_cmd(args):
    print("Running:", " ".join(args))
    result = subprocess.run(args, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)

cmd_memories = [
    "gcloud.cmd", "firestore", "indexes", "composite", "create",
    f"--project={os.environ.get('GOOGLE_CLOUD_PROJECT', '')}",
    "--collection-group=agent_memories",
    "--query-scope=COLLECTION",
    "--field-config=order=ASCENDING,field-path=agent_id",
    "--field-config=vector-config={\"dimension\":\"768\",\"flat\": {}},field-path=embedding"
]

cmd_board = [
    "gcloud.cmd", "firestore", "indexes", "composite", "create",
    f"--project={os.environ.get('GOOGLE_CLOUD_PROJECT', '')}",
    "--collection-group=board_posts",
    "--query-scope=COLLECTION",
    "--field-config=vector-config={\"dimension\":\"768\",\"flat\": {}},field-path=embedding"
]

print("Creating agent_memories index...")
run_cmd(cmd_memories)

print("Creating board_posts index...")
run_cmd(cmd_board)
