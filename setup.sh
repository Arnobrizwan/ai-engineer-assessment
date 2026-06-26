#!/usr/bin/env bash
#
# setup.sh — one entry point to bootstrap, test, and run all three projects.
#
#   ./setup.sh setup        Bootstrap everything: check tools, pull models,
#                           create per-project venvs, install deps, seed Q3 DB.
#   ./setup.sh test         Run every project's unit suite (no LLM needed).
#   ./setup.sh run q1       Launch Q1 Agentic RAG (Streamlit)      → :8501
#   ./setup.sh run q2       Launch Q2 Streaming Chat (FastAPI)     → :8000
#   ./setup.sh run q3       Launch Q3 Agentic AI / SQL (Streamlit) → :8502
#   ./setup.sh run all      Launch all three in the background
#   ./setup.sh stop         Stop apps started by "run all"
#   ./setup.sh doctor       Show tool / model / venv status
#
# Config (override via env): LLM_MODEL, EMBED_MODEL, Q1_PORT, Q2_PORT, Q3_PORT.
set -euo pipefail

# --- paths & config ----------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Q1="$ROOT/q1-agentic-rag"
Q2="$ROOT/q2-streaming-chat"
Q3="$ROOT/q3-agentic-ai"
RUN_DIR="$ROOT/.run"

LLM_MODEL="${LLM_MODEL:-llama3.1:8b}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
Q1_PORT="${Q1_PORT:-8501}"
Q2_PORT="${Q2_PORT:-8000}"
Q3_PORT="${Q3_PORT:-8502}"

# --- pretty output -----------------------------------------------------------
c_blue='\033[0;34m'; c_green='\033[0;32m'; c_yellow='\033[0;33m'; c_red='\033[0;31m'; c_off='\033[0m'
info()  { printf "${c_blue}==>${c_off} %s\n" "$*"; }
ok()    { printf "${c_green}  ✓${c_off} %s\n" "$*"; }
warn()  { printf "${c_yellow}  !${c_off} %s\n" "$*"; }
die()   { printf "${c_red}  ✗ %s${c_off}\n" "$*" >&2; exit 1; }

# --- helpers -----------------------------------------------------------------
pick_python() {
  # Prefer python3.11 (the version this repo was built/tested on), else python3.
  for c in python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1; then echo "$c"; return 0; fi
  done
  die "No python3 found. Install Python 3.11+."
}
PYTHON="$(pick_python)"

require_ollama() {
  command -v ollama >/dev/null 2>&1 || die "Ollama not found. Install from https://ollama.com"
  if ! curl -s --max-time 3 "http://localhost:11434/api/tags" >/dev/null 2>&1; then
    warn "Ollama daemon not reachable on :11434 — start it with 'ollama serve' (the apps need it at runtime)."
  fi
}

pull_models() {
  require_ollama
  for m in "$LLM_MODEL" "$EMBED_MODEL"; do
    if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$m\(:latest\)\?"; then
      ok "model present: $m"
    else
      info "pulling model: $m"
      ollama pull "$m"
    fi
  done
}

# Create a venv and install pinned deps for one project (idempotent).
bootstrap_project() {
  local dir="$1" name="$2"
  [ -d "$dir" ] || die "missing project dir: $dir"
  info "bootstrapping $name"
  if [ ! -x "$dir/.venv/bin/python" ]; then
    "$PYTHON" -m venv "$dir/.venv"
    ok "created venv"
  else
    ok "venv exists"
  fi
  "$dir/.venv/bin/python" -m pip install --quiet --upgrade pip
  "$dir/.venv/bin/python" -m pip install --quiet -r "$dir/requirements.txt"
  ok "deps installed"
}

run_tests() {
  local dir="$1" name="$2"
  info "tests: $name"
  ( cd "$dir" && ./.venv/bin/python -m pytest -q )
}

# --- commands ----------------------------------------------------------------
cmd_setup() {
  info "Python interpreter: $PYTHON ($($PYTHON --version 2>&1))"
  pull_models
  bootstrap_project "$Q1" "Q1 Agentic RAG"
  bootstrap_project "$Q2" "Q2 Streaming Chat"
  bootstrap_project "$Q3" "Q3 Agentic AI"
  info "seeding Q3 analytics database"
  ( cd "$Q3" && ./.venv/bin/python seed.py >/dev/null && ok "analytics.db ready" )
  printf "\n${c_green}Setup complete.${c_off} Next:\n"
  printf "  ./setup.sh test         # run all unit suites\n"
  printf "  ./setup.sh run q1       # or q2 / q3 / all\n"
}

cmd_test() {
  run_tests "$Q1" "Q1 Agentic RAG"
  run_tests "$Q2" "Q2 Streaming Chat"
  run_tests "$Q3" "Q3 Agentic AI"
  printf "\n${c_green}All unit suites passed.${c_off}\n"
}

ensure_ready() {
  # Fail early with a friendly hint if a project hasn't been bootstrapped.
  local dir="$1"
  [ -x "$dir/.venv/bin/python" ] || die "venv missing for $(basename "$dir") — run './setup.sh setup' first."
}

launch_q1() { ensure_ready "$Q1"; ( cd "$Q1" && exec ./.venv/bin/streamlit run app.py --server.port "$Q1_PORT" --server.headless true ); }
launch_q3() { ensure_ready "$Q3"; ( cd "$Q3" && ./.venv/bin/python seed.py >/dev/null 2>&1; exec ./.venv/bin/streamlit run app.py --server.port "$Q3_PORT" --server.headless true ); }
launch_q2() { ensure_ready "$Q2"; ( cd "$Q2" && exec ./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$Q2_PORT" ); }

cmd_run() {
  require_ollama
  case "${1:-}" in
    q1) info "Q1 Agentic RAG → http://localhost:$Q1_PORT (Ctrl-C to stop)"; launch_q1 ;;
    q2) info "Q2 Streaming Chat → http://localhost:$Q2_PORT (Ctrl-C to stop)"; launch_q2 ;;
    q3) info "Q3 Agentic AI → http://localhost:$Q3_PORT (Ctrl-C to stop)"; launch_q3 ;;
    all) cmd_run_all ;;
    *) die "usage: ./setup.sh run <q1|q2|q3|all>" ;;
  esac
}

# Background one app so that $! is the real process (exec replaces the subshell),
# recording its pid and port for a reliable stop.
_spawn() {
  local name="$1" dir="$2" port="$3"; shift 3
  ( cd "$dir" && exec "$@" ) >"$RUN_DIR/$name.log" 2>&1 &
  echo "$!"   >"$RUN_DIR/$name.pid"
  echo "$port" >"$RUN_DIR/$name.port"
}

cmd_run_all() {
  ensure_ready "$Q1"; ensure_ready "$Q2"; ensure_ready "$Q3"
  mkdir -p "$RUN_DIR"
  ( cd "$Q3" && ./.venv/bin/python seed.py >/dev/null 2>&1 ) || true
  info "starting all three in the background (logs in .run/)"
  _spawn q1 "$Q1" "$Q1_PORT" ./.venv/bin/streamlit run app.py --server.port "$Q1_PORT" --server.headless true
  _spawn q2 "$Q2" "$Q2_PORT" ./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$Q2_PORT"
  _spawn q3 "$Q3" "$Q3_PORT" ./.venv/bin/streamlit run app.py --server.port "$Q3_PORT" --server.headless true
  sleep 4
  ok "Q1 Agentic RAG    → http://localhost:$Q1_PORT"
  ok "Q2 Streaming Chat → http://localhost:$Q2_PORT"
  ok "Q3 Agentic AI     → http://localhost:$Q3_PORT"
  printf "\nTail logs:  tail -f .run/q*.log\nStop all:   ./setup.sh stop\n"
}

# Kill a process and any children it spawned (Streamlit/uvicorn fork workers).
_kill_tree() {
  local pid="$1"
  pkill -P "$pid" >/dev/null 2>&1 || true
  kill "$pid" >/dev/null 2>&1 || true
}

# Free a TCP port by killing whatever still listens on it (safety net).
_free_port() {
  local port="$1" pids
  pids="$(lsof -ti "tcp:$port" 2>/dev/null || true)"
  [ -n "$pids" ] && echo "$pids" | xargs kill >/dev/null 2>&1 || true
}

cmd_stop() {
  local stopped=0
  if [ -d "$RUN_DIR" ]; then
    for p in "$RUN_DIR"/*.pid; do
      [ -e "$p" ] || continue
      local name pid; name="$(basename "$p" .pid)"; pid="$(cat "$p")"
      _kill_tree "$pid"
      ok "stopped $name (pid $pid)"; stopped=1
      rm -f "$p"
    done
  fi
  # Safety net: free the configured ports even if children were re-parented.
  for port in "$Q1_PORT" "$Q2_PORT" "$Q3_PORT"; do _free_port "$port"; done
  rm -f "$RUN_DIR"/*.port 2>/dev/null || true
  [ "$stopped" -eq 1 ] || warn "no tracked apps; freed configured ports anyway"
}

cmd_doctor() {
  info "Tooling"
  printf "  python : %s\n" "$($PYTHON --version 2>&1)"
  printf "  ollama : %s\n" "$(command -v ollama >/dev/null 2>&1 && ollama --version 2>&1 || echo 'NOT FOUND')"
  printf "  docker : %s\n" "$(command -v docker >/dev/null 2>&1 && docker --version 2>&1 || echo 'not found (optional)')"
  info "Ollama models"
  if curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    ollama list 2>/dev/null | sed 's/^/  /'
  else
    warn "daemon not reachable (run 'ollama serve')"
  fi
  info "Project venvs"
  for d in "$Q1" "$Q2" "$Q3"; do
    [ -x "$d/.venv/bin/python" ] && ok "$(basename "$d"): ready" || warn "$(basename "$d"): not bootstrapped"
  done
}

usage() {
  # Print the header comment block (from line 3 until the first non-# line).
  awk 'NR>=3 { if (/^#/) { sub(/^# ?/, ""); print } else { exit } }' "${BASH_SOURCE[0]}"
}

main() {
  case "${1:-help}" in
    setup)  cmd_setup ;;
    test)   cmd_test ;;
    run)    shift; cmd_run "$@" ;;
    stop)   cmd_stop ;;
    doctor) cmd_doctor ;;
    help|-h|--help|"") usage ;;
    *) die "unknown command: $1 (try './setup.sh help')" ;;
  esac
}
main "$@"
