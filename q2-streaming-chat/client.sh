#!/usr/bin/env bash
#
# client.sh — minimal terminal client that streams a chat reply token-by-token
# from the FastAPI SSE endpoint, printing tokens as they arrive.
#
#   ./client.sh "your message" [session_id] [base_url]
#
# Same session_id => the server replays prior turns, so the model remembers.
set -euo pipefail

msg="${1:?usage: ./client.sh <message> [session_id] [base_url]}"
sid="${2:-cli}"
base="${3:-http://localhost:8000}"

payload="$(python3 -c 'import json,sys; print(json.dumps({"session_id":sys.argv[1],"message":sys.argv[2]}))' "$sid" "$msg")"

curl -sN -X POST "$base/api/chat" -H 'Content-Type: application/json' -d "$payload" \
| python3 -c '
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data:"):
        continue
    event = json.loads(line[5:])
    kind = event.get("type")
    if kind == "token":
        sys.stdout.write(event["content"]); sys.stdout.flush()
    elif kind == "done":
        print()
'
