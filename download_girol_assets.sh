#!/usr/bin/env bash
set -Eeuo pipefail

# Download and install GIROL assets.
#
# Place this script in the ROOT of the IsaacLab repository:
#
#   IsaacLab/
#   ├── download_girol_assets.sh
#   ├── source/
#   └── ...
#
# Run from anywhere:
#
#   ./download_girol_assets.sh
#
# Assets are installed into:
#
#   source/isaaclab_assets/data/aloha_assets

DRIVE_FOLDER_URL="https://drive.google.com/drive/folders/1U7mSc20WLV4vuRWFKpn2K5gKLYJFYJKw?usp=sharing"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET_PARENT="${REPO_ROOT}/source/isaaclab_assets/data"
TARGET_DIR="${TARGET_PARENT}/aloha_assets"

if [[ ! -d "${REPO_ROOT}/source/isaaclab_assets" ]]; then
    echo "Error: this script must be placed in the IsaacLab repository root." >&2
    echo "Expected directory not found:" >&2
    echo "  ${REPO_ROOT}/source/isaaclab_assets" >&2
    exit 1
fi

mkdir -p "$TARGET_DIR"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is not installed or is not available in PATH." >&2
    exit 1
fi

if ! command -v file >/dev/null 2>&1; then
    echo "Error: the 'file' utility is not installed." >&2
    exit 1
fi

if ! python3 -c "import gdown" >/dev/null 2>&1; then
    echo "[1/5] Installing gdown..."
    python3 -m pip install --upgrade gdown
else
    echo "[1/5] gdown is already installed."
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

DOWNLOAD_DIR="${TMP_DIR}/drive_folder"
EXTRACT_DIR="${TMP_DIR}/extracted"
mkdir -p "$DOWNLOAD_DIR" "$EXTRACT_DIR"

echo "[2/5] Downloading the GIROL Drive folder..."
python3 -m gdown \
    "$DRIVE_FOLDER_URL" \
    --folder \
    -O "$DOWNLOAD_DIR"

mapfile -d '' ZIP_FILES < <(
    find "$DOWNLOAD_DIR" -type f -iname '*.zip' -print0
)

if [[ ${#ZIP_FILES[@]} -eq 0 ]]; then
    echo "Error: no ZIP archive was found in the downloaded Drive folder." >&2
    echo "Downloaded files:" >&2
    find "$DOWNLOAD_DIR" -type f -printf '  %p\n' >&2
    exit 1
fi

if [[ ${#ZIP_FILES[@]} -gt 1 ]]; then
    echo "Error: more than one ZIP archive was found:" >&2
    printf '  %s\n' "${ZIP_FILES[@]}" >&2
    exit 1
fi

ZIP_FILE="${ZIP_FILES[0]}"
echo "[3/5] Found archive: $(basename "$ZIP_FILE")"

echo "[4/5] Extracting archive..."
python3 - "$ZIP_FILE" "$EXTRACT_DIR" <<'PY'
from pathlib import Path
import sys
import zipfile

archive_path = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()

with zipfile.ZipFile(archive_path) as archive:
    for member in archive.infolist():
        output_path = (destination / member.filename).resolve()
        try:
            output_path.relative_to(destination)
        except ValueError as exc:
            raise RuntimeError(
                f"Unsafe path in ZIP archive: {member.filename!r}"
            ) from exc

    archive.extractall(destination)
PY

# If the archive contains exactly one top-level directory, copy its contents
# rather than creating aloha_assets/<wrapper-directory>/...
mapfile -d '' TOP_LEVEL_ENTRIES < <(
    find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -print0
)

SOURCE_DIR="$EXTRACT_DIR"
if [[ ${#TOP_LEVEL_ENTRIES[@]} -eq 1 && -d "${TOP_LEVEL_ENTRIES[0]}" ]]; then
    SOURCE_DIR="${TOP_LEVEL_ENTRIES[0]}"
fi

echo "[5/5] Installing assets into:"
echo "      $TARGET_DIR"

if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 "$SOURCE_DIR"/ "$TARGET_DIR"/
else
    cp -a "$SOURCE_DIR"/. "$TARGET_DIR"/
fi

echo
echo "GIROL assets installed successfully."
echo "Destination: $TARGET_DIR"
du -sh "$TARGET_DIR" 2>/dev/null || true

echo
echo "Sample files:"
find "$TARGET_DIR" -maxdepth 3 -type f | head -n 20
