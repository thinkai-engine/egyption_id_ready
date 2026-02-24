#!/bin/bash
# ═══════════════════════════════════════════════════════
#  Egyptian ID OCR — Full Pipeline (one-shot)
# ═══════════════════════════════════════════════════════
# Usage:
#   bash scripts/run_full_pipeline.sh
# ═══════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

echo "════════════════════════════════════════"
echo "   Egyptian ID OCR — Full Pipeline"
echo "════════════════════════════════════════"

# 1. قص الحقول
echo ""
echo "📸 Step 1: Cropping fields..."
python scripts/build_dataset.py

# 2. استخراج النصوص
echo ""
echo "🤗 Step 2: Labeling with OCR..."
python scripts/label_crops.py --method qari

# 3. تجهيز ملفات PaddleOCR
echo ""
echo "📄 Step 3: Building PaddleOCR label files..."
python scripts/prepare_paddle_labels.py

# 4. Fine-tuning
echo ""
echo "🏋️  Step 4: Fine-tuning..."
bash scripts/train.sh

# 5. تقييم
echo ""
echo "📊 Step 5: Evaluating..."
python scripts/evaluate.py

# 6. تصدير ONNX
echo ""
echo "📦 Step 6: Exporting to ONNX..."
bash scripts/export_onnx.sh

# 7. Benchmark
echo ""
echo "⚡ Step 7: Benchmarking..."
python scripts/benchmark.py

# 8. Docker
echo ""
echo "🚀 Step 8: Starting API server..."
docker compose up --build -d

# 9. Tests
echo ""
echo "🧪 Step 9: Running tests..."
python -m pytest tests/test_pipeline.py -v

echo ""
echo "════════════════════════════════════════"
echo "✅ Pipeline complete!"
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo "════════════════════════════════════════"
