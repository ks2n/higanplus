#!/usr/bin/env bash
# Download IAM dataset (h5py) + pretrained checkpoints from upstream releases.
# Idempotent: skips files that already exist with the right size.
#
# Usage:  bash scripts/setup_data.sh
# Source: https://github.com/ganji15/HiGANplus/releases/tag/dataset

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/HiGAN+"

mkdir -p data/iam pretrained

# -------- IAM dataset --------
DATA_BASE="https://github.com/ganji15/HiGANplus/releases/download/dataset"

download() {
    local url="$1"
    local dst="$2"
    if [[ -f "$dst" && -s "$dst" ]]; then
        echo "[skip] $dst already exists ($(du -h "$dst" | cut -f1))"
        return
    fi
    echo "[get ] $url -> $dst"
    curl -L --fail --retry 3 -o "$dst" "$url"
}

download "$DATA_BASE/trnvalset_words64_OrgSz.hdf5" "data/iam/trnvalset_words64_OrgSz.hdf5"
download "$DATA_BASE/testset_words64_OrgSz.hdf5"   "data/iam/testset_words64_OrgSz.hdf5"

# -------- Pretrained checkpoints --------
# These are released alongside the dataset on the same upstream release page.
# If the upstream release URL changes, update CKPT_BASE accordingly.
CKPT_BASE="https://github.com/ganji15/HiGANplus/releases/download/dataset"

download "$CKPT_BASE/deploy_HiGAN+.pth"  "pretrained/deploy_HiGAN+.pth"
download "$CKPT_BASE/ocr_iam_new.pth"    "pretrained/ocr_iam_new.pth"
download "$CKPT_BASE/wid_iam_new.pth"    "pretrained/wid_iam_new.pth"
download "$CKPT_BASE/wid_iam_test.pth"   "pretrained/wid_iam_test.pth"

echo
echo "Done. Files:"
ls -lh data/iam/*.hdf5 pretrained/*.pth
