#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage(){
  cat <<EOF
Usage: $0 [--remove-images] [--remove-volumes] [--prune] [--force]

Options:
  --remove-images   Remove the local image tag 'agent-bci-llm:latest' after stopping
  --remove-volumes  Remove named docker volumes declared in compose (may delete data)
  --prune           Run docker system prune -f (removes dangling images/containers)
  --force           Skip confirmation prompts
EOF
  exit 1
}

REMOVE_IMAGES=false
REMOVE_VOLUMES=false
PRUNE=false
FORCE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remove-images) REMOVE_IMAGES=true; shift ;;
    --remove-volumes) REMOVE_VOLUMES=true; shift ;;
    --prune) PRUNE=true; shift ;;
    --force) FORCE=true; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

confirm(){
  if $FORCE; then
    return 0
  fi
  read -r -p "$1 [y/N]: " ans
  case "$ans" in
    [yY]|[yY][eE][sS]) return 0 ;;
    *) return 1 ;;
  esac
}

echo "[agent] Bringing down docker-compose services (bci)"
docker compose down || true

if $REMOVE_IMAGES; then
  if confirm "Remove image agent-bci-llm:latest?"; then
    echo "[agent] Removing image agent-bci-llm:latest"
    docker image rm agent-bci-llm:latest || true
  fi
fi

if $REMOVE_VOLUMES; then
  if confirm "Remove docker-compose named volumes? (destructive)"; then
    echo "[agent] Removing named volumes from compose"
    docker compose down -v || true
  fi
fi

if $PRUNE; then
  if confirm "Run 'docker system prune -f' to remove unused data?"; then
    docker system prune -f || true
  fi
fi

echo "[agent] Done. Use 'docker compose ps' and 'docker images' to verify state."
