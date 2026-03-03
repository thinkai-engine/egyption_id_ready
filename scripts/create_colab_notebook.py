#!/usr/bin/env python3
"""
Create Colab Notebook for Egyptian ID OCR Project
Generates a complete Google Colab notebook that runs the full pipeline.
"""

import json
from pathlib import Path

notebook = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# 🇪🇬 Egyptian ID OCR - Complete Google Colab Notebook\n",
        "\n",
        "This notebook runs the **complete Egyptian ID OCR pipeline** on Google Colab with GPU support.\n",
        "\n",
        "## 📋 What This Notebook Does\n",
        "\n",
        "1. **Setup Environment** - Install all dependencies\n",
        "2. **Download Models** - YOLO detection models + OCR models from direct links\n",
        "3. **Download Dataset** - Egyptian ID dataset from Google Drive\n",
        "4. **Build Dataset** - Crop fields from ID card images using two-stage detection\n",
        "5. **Label Crops** - Extract text using QARI/Bakri/AirLLM OCR engines\n",
        "6. **Train PaddleOCR** - Fine-tune OCR model on Egyptian ID data\n",
        "7. **Evaluate & Export** - Calculate accuracy metrics and export to ONNX\n",
        "8. **Deploy API** - Test the FastAPI endpoint\n",
        "\n",
        "## ⚙️ Runtime Settings\n",
        "\n",
        "**Recommended Colab Settings:**\n",
        "- **Runtime Type**: GPU (T4 or better)\n",
        "- **RAM**: High-RAM runtime if available\n",
        "- **Storage**: Files will be stored in /content/egyption_id_ready/\n",
        "\n",
        "**Estimated Runtime:**\n",
        "- Setup + Downloads: 15-20 minutes\n",
        "- Dataset Processing: 5-10 minutes\n",
        "- OCR Labeling: 30-60 minutes (depends on method)\n",
        "- Training: 2-4 hours (100 epochs)\n",
        "\n",
        "## 🔗 Quick Links\n",
        "\n",
        "- **Dataset**: Google Drive (auto-downloaded)\n",
        "- **YOLO Models**: GitHub NASO7Y repository (auto-downloaded)\n",
        "- **OCR Models**: HuggingFace + EasyOCR (auto-downloaded)\n",
        "- **Project Repo**: https://github.com/your-repo/egyption_id_ready"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 📦 Part 1: Environment Setup\n",
        "\n",
        "### 1.1 Check GPU and Mount Google Drive"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Check GPU\n",
        "!nvidia-smi\n",
        "\n",
        "# Check Python version\n",
        "!python --version\n",
        "\n",
        "# Mount Google Drive (optional - for saving outputs)\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive', force_remount=False)\n",
        "\n",
        "print(\"✅ Environment check complete!\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 1.2 Clone Repository and Install Dependencies"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Clone the repository\n",
        "%cd /content\n",
        "!git clone https://github.com/your-repo/egyption_id_ready.git\n",
        "%cd /content/egyption_id_ready\n",
        "\n",
        "# Install system dependencies\n",
        "!apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libgomp1 curl wget git\n",
        "\n",
        "print(\"✅ Repository cloned and system dependencies installed!\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Install Python dependencies\n",
        "!pip install -q -r requirements.txt\n",
        "!pip install -q google-generativeai\n",
        "\n",
        "print(\"✅ Python dependencies installed!\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 1.3 Verify Installation"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import torch, cv2, pandas as pd, numpy as np\n",
        "\n",
        "print(f\"PyTorch: {torch.__version__}\")\n",
        "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
        "print(f\"OpenCV: {cv2.__version__}\")\n",
        "print(f\"Pandas: {pd.__version__}\")\n",
        "\n",
        "if torch.cuda.is_available():\n",
        "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
        "    print(f\"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")\n",
        "\n",
        "print(\"\\n✅ Installation verified!\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 📥 Part 2: Download Models\n",
        "\n",
        "### 2.1 Download YOLO Detection Models from GitHub"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!mkdir -p /content/egyption_id_ready/weights\n",
        "\n",
        "print(\"⬇️ Downloading Card Detection Model...\")\n",
        "!wget -q --show-progress -O /content/egyption_id_ready/weights/card_detection.pt https://github.com/NASO7Y/OCR_Egyptian_ID/raw/main/detect_id_card.pt\n",
        "\n",
        "print(\"\\n⬇️ Downloading Field Detection Model...\")\n",
        "!wget -q --show-progress -O /content/egyption_id_ready/weights/field_detection.pt https://github.com/NASO7Y/OCR_Egyptian_ID/raw/main/detect_odjects.pt\n",
        "\n",
        "print(\"\\n⬇️ Downloading NID Detection Model...\")\n",
        "!wget -q --show-progress -O /content/egyption_id_ready/weights/nid_detection.pt https://github.com/NASO7Y/OCR_Egyptian_ID/raw/main/detect_id.pt\n",
        "\n",
        "print(\"\\n📦 Model Files:\")\n",
        "!ls -lh /content/egyption_id_ready/weights/"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 2.2 Download OCR Models"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"⬇️ Preparing EasyOCR models (Arabic + English)...\")\n",
        "!mkdir -p /content/egyption_id_ready/models_cache\n",
        "\n",
        "import easyocr\n",
        "reader = easyocr.Reader(['ar', 'en'], gpu=True, model_storage_directory='/content/egyption_id_ready/models_cache')\n",
        "print(\"✅ EasyOCR initialized!\")\n",
        "\n",
        "print(\"\\n⬇️ Downloading PaddleOCR pretrained model...\")\n",
        "!mkdir -p /content/egyption_id_ready/arabic_PP-OCRv3_rec\n",
        "!wget -q --show-progress -O /content/egyption_id_ready/arabic_PP-OCRv3_rec/best_accuracy.pdparams https://paddleocr.bj.bcebos.com/PP-OCRv3/arabic/rec_arabic_ppocr_v3_train/best_accuracy.pdparams\n",
        "\n",
        "print(\"\\n✅ OCR models ready!\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 📊 Part 3: Download Dataset\n",
        "\n",
        "### 3.1 Download Egyptian ID Dataset from Google Drive\n",
        "\n",
        "**Dataset Link**: Replace with your Google Drive link\n",
        "\n",
        "The dataset should contain:\n",
        "- `train/images/` - Training images + labels\n",
        "- `valid/images/` - Validation images + labels  \n",
        "- `test/images/` - Test images + labels"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "FILE_ID = \"YOUR_DATASET_FILE_ID_HERE\"\n",
        "OUTPUT_PATH = \"/content/egyption_id_ready/dataset.zip\"\n",
        "\n",
        "print(f\"⬇️ Downloading dataset from Google Drive...\")\n",
        "!pip install -q gdown\n",
        "!gdown --id {FILE_ID} -O {OUTPUT_PATH}\n",
        "\n",
        "print(\"\\n✅ Dataset downloaded!\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"📦 Extracting dataset...\")\n",
        "!unzip -q /content/egyption_id_ready/dataset.zip -d /content/egyption_id_ready/\n",
        "\n",
        "print(\"\\n📊 Dataset Statistics:\")\n",
        "for split in ['train', 'valid', 'test']:\n",
        "    img_count = !ls /content/egyption_id_ready/{split}/images/*.jpg 2>/dev/null | wc -l\n",
        "    lbl_count = !ls /content/egyption_id_ready/{split}/labels/*.txt 2>/dev/null | wc -l\n",
        "    print(f\"   {split:6s}: {img_count[0]:>6} images, {lbl_count[0]:>6} labels\")\n",
        "\n",
        "print(\"\\n✅ Dataset ready!\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 🔧 Part 4: Build Dataset with Two-Stage Detection"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import sys\n",
        "from pathlib import Path\n",
        "ROOT = Path('/content/egyption_id_ready')\n",
        "sys.path.insert(0, str(ROOT))\n",
        "print(f\"📂 Project root: {ROOT}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import cv2, matplotlib.pyplot as plt\n",
        "from src.card_detector import load_card_detector\n",
        "\n",
        "print(\"🔧 Loading two-stage detector...\")\n",
        "detector = load_card_detector(\n",
        "    card_model_path=str(ROOT / 'weights' / 'card_detection.pt'),\n",
        "    field_model_path=str(ROOT / 'weights' / 'field_detection.pt'),\n",
        ")\n",
        "print(\"✅ Detector loaded!\")\n",
        "\n",
        "test_images = list((ROOT / 'train' / 'images').glob('*.jpg'))\n",
        "if test_images:\n",
        "    image = cv2.imread(str(test_images[0]))\n",
        "    card_crop, fields = detector.detect_full(image, translate_to_project=True)\n",
        "    print(f\"✂️  Card crop: {card_crop.shape[1]}x{card_crop.shape[0]}\")\n",
        "    print(f\"📋 Fields detected: {len(fields)}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"🚀 Processing full dataset...\")\n",
        "!python /content/egyption_id_ready/scripts/process_full_dataset_two_stage.py --splits train valid test\n",
        "print(\"\\n✅ Dataset processing complete!\")\n",
        "!ls -lh /content/egyption_id_ready/rec/images/two_stage/ | head -5"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 🏷️ Part 5: Label Crops with OCR"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "OCR_METHOD = \"qari-airllm\"\n",
        "print(f\"🏷️ Labeling crops with {OCR_METHOD}...\")\n",
        "\n",
        "!python /content/egyption_id_ready/scripts/label_crops.py --method {OCR_METHOD} --splits train valid\n",
        "\n",
        "print(\"\\n✅ Labeling complete!\")\n",
        "!wc -l /content/egyption_id_ready/crops_labeled.csv"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 🎯 Part 6: Prepare Training Data"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"📝 Preparing training data...\")\n",
        "!python /content/egyption_id_ready/scripts/prepare_paddle_labels.py --input /content/egyption_id_ready/crops_labeled.csv --output-dir /content/egyption_id_ready/rec\n",
        "print(\"\\n✅ Training data prepared!\")\n",
        "!head -3 /content/egyption_id_ready/rec/train.txt"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 🚀 Part 7: Train PaddleOCR Model"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"🚀 Starting PaddleOCR training (2-4 hours)...\")\n",
        "!cd /content/egyption_id_ready && python PaddleOCR/tools/train.py -c configs/egyptian_id_rec.yml -o Global.use_gpu=true -o Global.epoch_num=100\n",
        "print(\"\\n✅ Training complete!\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 📊 Part 8: Evaluate Model"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"📊 Evaluating model...\")\n",
        "!python /content/egyption_id_ready/scripts/evaluate.py --model-path /content/egyption_id_ready/output/egyptian_id_rec/best_accuracy --test-data /content/egyption_id_ready/rec/test.txt\n",
        "print(\"\\n✅ Evaluation complete!\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 📦 Part 9: Export to ONNX"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!mkdir -p /content/egyption_id_ready/onnx\n",
        "!python /content/egyption_id_ready/PaddleOCR/tools/export_model.py -c /content/egyption_id_ready/configs/egyptian_id_rec.yml -o Global.pretrained_model=/content/egyption_id_ready/output/egyptian_id_rec/best_accuracy\n",
        "!paddle2onnx --model_dir /content/egyption_id_ready/inference/rec --save_file /content/egyption_id_ready/onnx/rec.onnx --opset_version 11\n",
        "!python -m onnxsim /content/egyption_id_ready/onnx/rec.onnx /content/egyption_id_ready/onnx/rec_sim.onnx\n",
        "print(\"\\n✅ ONNX export complete!\")\n",
        "!ls -lh /content/egyption_id_ready/onnx/"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 🔌 Part 10: Test Inference"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "sys.path.insert(0, '/content/egyption_id_ready')\n",
        "from src.inference import EgyptianIDOCR\n",
        "\n",
        "ocr = EgyptianIDOCR(\n",
        "    det_onnx=str(ROOT / 'model' / 'field_detector.onnx'),\n",
        "    rec_onnx=str(ROOT / 'onnx' / 'rec_sim.onnx'),\n",
        "    dict_path=str(ROOT / 'arabic_dict.txt'),\n",
        "    use_gpu=True,\n",
        ")\n",
        "print(\"✅ OCR loaded!\")\n",
        "\n",
        "test_images = list((ROOT / 'test' / 'images').glob('*.jpg'))\n",
        "if test_images:\n",
        "    image = cv2.imread(str(test_images[0]))\n",
        "    results = ocr.extract(image)\n",
        "    for field_name, field_data in results.items():\n",
        "        print(f\"{field_name}: {field_data.get('text', 'N/A')}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 💾 Part 11: Save to Google Drive"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!mkdir -p /content/drive/MyDrive/egyption_id_ready/output\n",
        "!cp -r /content/egyption_id_ready/output/egyptian_id_rec/ /content/drive/MyDrive/egyption_id_ready/output/\n",
        "!cp -r /content/egyption_id_ready/onnx/ /content/drive/MyDrive/egyption_id_ready/\n",
        "!cp /content/egyption_id_ready/crops_labeled.csv /content/drive/MyDrive/egyption_id_ready/\n",
        "print(\"\\n✅ All outputs saved to Google Drive!\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 📊 Summary"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"=\" * 60)\n",
        "print(\"🎉 EGYPTIAN ID OCR - COLAB NOTEBOOK COMPLETE!\")\n",
        "print(\"=\" * 60)\n",
        "print(\"\\n✅ Accomplished:\")\n",
        "print(\"   1. Environment setup with GPU\")\n",
        "print(\"   2. Downloaded YOLO + OCR models\")\n",
        "print(\"   3. Processed dataset\")\n",
        "print(\"   4. Labeled crops with OCR\")\n",
        "print(\"   5. Trained PaddleOCR\")\n",
        "print(\"   6. Evaluated model\")\n",
        "print(\"   7. Exported to ONNX\")\n",
        "print(\"   8. Tested inference\")\n",
        "print(\"   9. Saved to Google Drive\")\n",
        "print(\"\\n📂 Outputs:\")\n        print(f\"   - Model: /content/egyption_id_ready/output/\")\n        print(f\"   - ONNX: /content/egyption_id_ready/onnx/\")\n        print(f\"   - Drive: /content/drive/MyDrive/egyption_id_ready/\")\n        print(\"=\" * 60)"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}

if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "notebooks" / "Egyptian_ID_OCR_Full_Colab.ipynb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ Colab notebook created successfully!")
    print(f"   Location: {output_path}")
    print("\n📋 To use this notebook:")
    print("   1. Upload to Google Colab or open in Jupyter")
    print("   2. Replace YOUR_DATASET_FILE_ID_HERE with actual Google Drive file ID")
    print("   3. Update git clone URL to your repository")
    print("   4. Run all cells sequentially")
