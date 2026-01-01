#!/usr/bin/env bash
set -euo pipefail
# Interactive install helper:
# 1) Starts the bci service (no heavy build)
# 2) Execs into the container for manual installs
# 3) After user exits, commits the container to agent-bci-llm:latest

WORKDIR=$(pwd)
COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.yml}
SERVICE=${1:-bci}

echo "Starting compose service (build skipped for heavy installs)..."
docker compose up -d --build $SERVICE
CONTAINER_ID=$(docker compose ps -q $SERVICE)
if [ -z "$CONTAINER_ID" ]; then
  echo "Failed to get container id for $SERVICE"
  exit 1
fi

echo "Attaching to container $CONTAINER_ID. Install packages interactively now."
docker exec -it $CONTAINER_ID /bin/bash

read -p "Press Enter to commit the container to image agent-bci-llm:latest (or Ctrl-C to cancel)" dummy

echo "Committing container $CONTAINER_ID to agent-bci-llm:latest"
docker commit $CONTAINER_ID agent-bci-llm:latest

echo "Done. You can now use 'docker compose up -d --force-recreate bci' to start the baked image." 
