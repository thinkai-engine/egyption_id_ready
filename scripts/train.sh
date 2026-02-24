#!/bin/bash
# ═══════════════════════════════════════════════════
#  Egyptian ID OCR — Fine-tune PaddleOCR
# ═══════════════════════════════════════════════════
# Usage:
#   bash scripts/train.sh
#
# Prerequisites:
#   pip install paddlepaddle paddleocr
#   git clone https://github.com/PaddlePaddle/PaddleOCR
# ═══════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="$ROOT/configs/egyptian_id_rec.yml"

echo "════════════════════════════════════"
echo "  🏋️  PaddleOCR Fine-tuning"
echo "════════════════════════════════════"

# ── Verify prerequisites ──────────────────────────
if ! python -c "import paddle" 2>/dev/null; then
    echo "❌ paddlepaddle not installed!"
    echo "   Run: pip install paddlepaddle"
    exit 1
fi

if [ ! -f "$ROOT/rec/train.txt" ]; then
    echo "❌ Training labels not found!"
    echo "   Run: python scripts/prepare_paddle_labels.py"
    exit 1
fi

# ── Count training samples ────────────────────────
TRAIN_COUNT=$(wc -l < "$ROOT/rec/train.txt")
VAL_COUNT=$(wc -l < "$ROOT/rec/val.txt")
echo "  Train : $TRAIN_COUNT samples"
echo "  Val   : $VAL_COUNT samples"
echo "  Config: $CONFIG"
echo "  GPU   : false (CPU training)"
echo "════════════════════════════════════"

# ── Download pretrained model if missing ──────────
PRETRAINED="$ROOT/arabic_PP-OCRv3_rec_train"
if [ ! -d "$PRETRAINED" ]; then
    echo "⬇️  Downloading Arabic PP-OCRv3 pretrained model..."
    cd "$ROOT"
    wget -q https://paddleocr.bj.bcebos.com/PP-OCRv3/arabic/arabic_PP-OCRv3_rec_train.tar
    tar -xf arabic_PP-OCRv3_rec_train.tar
    rm arabic_PP-OCRv3_rec_train.tar
    echo "✅ Pretrained model ready"
fi

# ── Train ─────────────────────────────────────────
cd "$ROOT"
python tools/train.py -c "$CONFIG"

echo ""
echo "════════════════════════════════════"
echo "  ✅ Training complete!"
echo "  Model saved → output/egyptian_id_rec/"
echo "════════════════════════════════════"
