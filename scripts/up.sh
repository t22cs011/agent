#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<EOF
Usage: $0 [--build] [--pull-model] [--attach]

Options:
  --build       Build `agent-bci-llm:latest` with heavy deps (may require lots of RAM)
  --pull-model  Pull the Ollama model `deepseek-r1:14b` on the host before starting
  --attach      After starting, exec into the bci container and run `python3 /app/main.py`
EOF
  exit 1
}

BUILD=false
PULL_MODEL=false
ATTACH=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) BUILD=true; shift ;;
    --pull-model) PULL_MODEL=true; shift ;;
    --attach) ATTACH=true; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

echo "[agent] Starting quick-up script from $ROOT_DIR"

if $PULL_MODEL; then
  if command -v ollama >/dev/null 2>&1; then
    echo "[agent] Pulling model deepseek-r1:14b"
    sudo ollama pull deepseek-r1:14b || true
    ollama list
  else
    echo "[agent] Warning: `ollama` CLI not found in PATH. Install ollama on host or skip --pull-model."
  fi
fi

if $BUILD; then
  echo "[agent] Building image agent-bci-llm:latest (this can be slow and memory heavy)"
  docker build \
    --build-arg INSTALL_SCIENCE_PACKAGES=true \
    --build-arg INSTALL_LLM_PACKAGES=true \
    -t agent-bci-llm:latest .
fi

echo "[agent] Bringing up docker-compose services (bci)"
docker compose up -d --force-recreate bci

echo "[agent] Waiting 3s for services to initialize..."
sleep 3

echo "[agent] bci container status:"
docker compose ps

echo "[agent] Tail last 200 lines of bci logs (press Ctrl-C to stop):"
docker logs --tail 200 $(docker compose ps -q bci) || true

if $ATTACH; then
  container=$(docker compose ps -q bci)
  if [[ -n "$container" ]]; then
    echo "[agent] Attaching and running /app/main.py inside container $container"
    docker exec -it "$container" /bin/bash -c "python3 /app/main.py"
  else
    echo "[agent] No bci container found to attach"
  fi
fi

echo "[agent] Done. If Ollama is bound to 127.0.0.1 on the host and container cannot reach it, consider using host networking or forwarding." 
