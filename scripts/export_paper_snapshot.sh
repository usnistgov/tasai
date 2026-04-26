#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/export_paper_snapshot.sh /path/to/output-dir

Creates a clean, minimal TAS-AI snapshot for manuscript bundles. The snapshot
contains only the installable package plus a provenance manifest.
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1")"
SNAPSHOT_DIR="${OUT_DIR}/tasai"

cd "$REPO_ROOT"

GIT_SHA="$(git rev-parse HEAD)"
if GIT_TAG="$(git describe --tags --exact-match "$GIT_SHA" 2>/dev/null)"; then
  :
else
  GIT_TAG=""
fi

mkdir -p "$OUT_DIR"
rm -rf "$SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR"

rsync -a \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "$REPO_ROOT/tasai/" "$SNAPSHOT_DIR/tasai/"

cp "$REPO_ROOT/pyproject.toml" "$SNAPSHOT_DIR/"
cp "$REPO_ROOT/README.md" "$SNAPSHOT_DIR/"
cp "$REPO_ROOT/LICENSE" "$SNAPSHOT_DIR/"

cat > "$SNAPSHOT_DIR/SNAPSHOT_PROVENANCE.json" <<EOF
{
  "source_repo": "git@github.com:williamratcliff-nist/tasai.git",
  "source_commit": "$GIT_SHA",
  "source_tag": ${GIT_TAG:+"\"$GIT_TAG\""},
  "snapshot_created_utc": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "contents": [
    "tasai/",
    "pyproject.toml",
    "README.md",
    "LICENSE"
  ]
}
EOF

if [[ -z "$GIT_TAG" ]]; then
  perl -0pi -e 's/"source_tag": \s*,/"source_tag": null,/g' "$SNAPSHOT_DIR/SNAPSHOT_PROVENANCE.json"
fi

echo "Created clean TAS-AI snapshot at: $SNAPSHOT_DIR"
