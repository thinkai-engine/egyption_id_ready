#!/bin/bash
# ═══════════════════════════════════════════════════
#  Egyptian ID OCR — Export to ONNX
# ═══════════════════════════════════════════════════
# Usage:
#   bash scripts/export_onnx.sh
#
# Prerequisites:
#   pip install paddle2onnx onnxsim
# ═══════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="$ROOT/configs/egyptian_id_rec.yml"
BEST_MODEL="$ROOT/output/egyptian_id_rec/best_accuracy"
INFERENCE_DIR="$ROOT/inference/rec"
ONNX_DIR="$ROOT/onnx"

echo "════════════════════════════════════"
echo "  📦 Export PaddleOCR → ONNX"
echo "════════════════════════════════════"

# ── 1. Export Paddle inference model ──────────────
if [ ! -d "$INFERENCE_DIR" ]; then
    echo "  Step 1: Exporting inference model..."
    python tools/export_model.py \
        -c "$CONFIG" \
        -o Global.pretrained_model="$BEST_MODEL" \
           Global.save_inference_dir="$INFERENCE_DIR"
    echo "  ✅ Inference model → $INFERENCE_DIR"
else
    echo "  ⏭️  Inference model already exists"
fi

# ── 2. Convert to ONNX ───────────────────────────
mkdir -p "$ONNX_DIR"
echo "  Step 2: Converting to ONNX..."
paddle2onnx \
    --model_dir "$INFERENCE_DIR" \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file "$ONNX_DIR/rec.onnx" \
    --opset_version 11

echo "  ✅ ONNX model → $ONNX_DIR/rec.onnx"

# ── 3. Simplify / optimize ───────────────────────
echo "  Step 3: Optimizing with onnxsim..."
python -m onnxsim "$ONNX_DIR/rec.onnx" "$ONNX_DIR/rec_sim.onnx"
echo "  ✅ Optimized → $ONNX_DIR/rec_sim.onnx"

# ── 4. Report sizes ──────────────────────────────
ORIG_SIZE=$(du -h "$ONNX_DIR/rec.onnx" | cut -f1)
SIM_SIZE=$(du -h "$ONNX_DIR/rec_sim.onnx" | cut -f1)

echo ""
echo "════════════════════════════════════"
echo "  📊 Export Summary"
echo "    Original  : $ORIG_SIZE"
echo "    Optimized : $SIM_SIZE"
echo "    Location  : $ONNX_DIR/rec_sim.onnx"
echo "════════════════════════════════════"
echo "  ✅ Ready for deployment!"
