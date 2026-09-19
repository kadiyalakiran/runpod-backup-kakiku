#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="https://github.com/kadiyalakiran/runpod-backup-kakiku.git"
REPO_DIR="$WORKSPACE/runpod-backup-kakiku"
VENV_DIR="$WORKSPACE/.venv"

echo "==> Repo"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "==> Python venv"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install -q -U pip
pip install -q -r "$REPO_DIR/requirements.txt"

echo "==> Env vars"
export HF_HOME="$WORKSPACE/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"
if [ -f "$REPO_DIR/.env" ]; then
  set -a; source "$REPO_DIR/.env"; set +a
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "    WARNING: ANTHROPIC_API_KEY not set."
fi

echo "==> Claude Code CLI"
if ! command -v claude &> /dev/null; then
  if curl -fsSL https://claude.ai/install.sh | bash; then
    echo "    Installed."
  else
    echo "    WARNING: install failed — continuing anyway."
  fi
else
  echo "    Already installed."
fi

echo "==> VS Code extensions"
if command -v code &> /dev/null; then
  while IFS= read -r ext; do
    [ -z "$ext" ] && continue
    code --install-extension "$ext" --force > /dev/null 2>&1 || echo "    Failed: $ext"
  done < "$REPO_DIR/extensions.txt"
else
  echo "    'code' CLI not found yet — open a file from VS Code first, then re-run."
fi

echo ""
echo "==> Done. Next: source $VENV_DIR/bin/activate && claude doctor"
