#!/usr/bin/env bash
set -euo pipefail

REPO="${GITHUB_REPO:-Khoaph1709/Object-detection-from-scratch-}"
TAG="${GITHUB_RELEASE_TAG:-hem-final}"
ASSET_NAME="${GITHUB_RELEASE_ASSET:-hem-best.pth}"
SOURCE_CHECKPOINT="${1:-}"

if [[ -z "$SOURCE_CHECKPOINT" ]]; then
  echo "Usage: $0 /absolute/path/to/final_hem_checkpoint.pth" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $SOURCE_CHECKPOINT" >&2
  exit 1
fi
if [[ "${SOURCE_CHECKPOINT##*.}" != "pth" ]]; then
  echo "ERROR: checkpoint must have .pth extension: $SOURCE_CHECKPOINT" >&2
  exit 1
fi

command -v gh >/dev/null 2>&1 || {
  echo "ERROR: GitHub CLI (gh) is required." >&2
  exit 1
}
gh auth status >/dev/null

if ! gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  gh release create "$TAG" \
    --repo "$REPO" \
    --title "HEM final checkpoint" \
    --notes "Public inference checkpoint for the HEM final submission."
fi

gh release upload "$TAG" "$SOURCE_CHECKPOINT#$ASSET_NAME" --repo "$REPO" --clobber

URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET_NAME}"
SHA256="$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')"
printf 'Release asset uploaded successfully.\n'
printf 'URL: %s\n' "$URL"
printf 'SHA256: %s\n' "$SHA256"
printf 'Tag: %s\n' "$TAG"
printf 'Asset: %s\n' "$ASSET_NAME"
