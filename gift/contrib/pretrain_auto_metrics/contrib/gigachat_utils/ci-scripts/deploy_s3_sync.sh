#!/bin/bash
set -euo pipefail

echo "Deploying s3-synchronize on remote server..."

# Ensure required environment variables are set.
: "${REMOTE_TARGET_DIR:?}"
: "${REMOTE_HOST:?}"
: "${REMOTE_USER:?}"
: "${REMOTE_PORT:?}"
: "${SSH_KEY:?}"

echo "Syncing local s3-synchronize directory to remote..."
scp -r -P "$REMOTE_PORT" -i "$SSH_KEY" "$CI_PROJECT_DIR/s3-synchronize/." "$REMOTE_USER@$REMOTE_HOST:${REMOTE_TARGET_DIR}/"

echo "SSH into s3-synchronize-bot-test-v2 to update and restart..."
ssh $REMOTE_USER@$REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY <<EOF
  set -euo pipefail
  TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
  if [ -d "${REMOTE_TARGET_DIR}" ]; then
      cp -r "${REMOTE_TARGET_DIR}" "${REMOTE_TARGET_DIR}-tmp-copy-\$TIMESTAMP"
  fi
  cp -a "${REMOTE_TARGET_DIR}" "${REMOTE_TARGET_DIR}"
  if [ -f "${REMOTE_TARGET_DIR}/.env" ]; then
      source "${REMOTE_TARGET_DIR}/.env"
  fi
  tmux kill-session -t s3-synchronize-nfs1 || true
  tmux kill-session -t s3-synchronize-nfs2 || true
  tmux new-session -d -s s3-synchronize-nfs1 "cd ${REMOTE_TARGET_DIR} && python s3-synchronize.py /home/jovyan/gigachat_checkpoints gigachat_checkpoints --n_last_left 3 --sft_n_last_left 128 --clear_after 336 --sft_clear_after 48 --sleep_interval 180"
  tmux new-session -d -s s3-synchronize-nfs2 "cd ${REMOTE_TARGET_DIR} && python s3-synchronize.py /home/jovyan/shares/SR004.nfs2/gigachat_checkpoints gigachat_checkpoints/nfs2 --n_last_left 3 --sft_n_last_left 128 --clear_after 336 --sft_clear_after 48 --sleep_interval 180 --signal_dir /home/jovyan/alert-signals/s3-upload-nfs2"
EOF

echo "s3-synchronize deployment completed."
