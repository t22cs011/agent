#!/usr/bin/env bash
# Run the training script, retrying on segmentation fault (exit code 139) or killed (137).
# Usage: MAX_RETRIES=20 ./run_with_retries.sh

MAX_RETRIES=${MAX_RETRIES:-20}
COUNT=0
SCRIPT="python3 bci_code/train_cv_full.py"

while true; do
  ((COUNT++))
  echo "[run] Attempt $COUNT/$MAX_RETRIES"
  $SCRIPT
  EXIT=$?
  if [ $EXIT -eq 0 ]; then
    echo "[run] Completed successfully"
    exit 0
  fi
  # Retry on segmentation fault (139) or killed (137, often OOM)
  if [ $EXIT -ne 139 ] && [ $EXIT -ne 137 ]; then
    echo "[run] Exited with code $EXIT (not a retriable crash). Stopping."
    exit $EXIT
  fi
  if [ $COUNT -ge $MAX_RETRIES ]; then
    echo "[run] Reached max retries ($MAX_RETRIES). Stopping with exit $EXIT."
    exit $EXIT
  fi
  if [ $COUNT -lt 6 ]; then
    BACKOFF=$((2 ** COUNT))
  else
    BACKOFF=64
  fi
  echo "[run] Crash detected (exit $EXIT). Retrying in ${BACKOFF}s..."
  sleep $BACKOFF
done
