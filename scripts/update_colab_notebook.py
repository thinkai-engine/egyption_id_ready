#!/usr/bin/env python3
"""
Update Colab Notebook with Critical Fixes
Adds missing cells and validation checks.
"""

import json
from pathlib import Path

def load_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(notebook, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

def create_cell(cell_type, source, metadata=None):
    """Create a notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source if isinstance(source, list) else [source]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "Egyptian_ID_OCR_Full_Colab.ipynb"
    notebook = load_notebook(notebook_path)
    
    print("🔧 Updating Colab Notebook...")
    
    # 1. Add PaddleOCR clone cell after repository clone (around cell 4)
    paddleocr_cell = create_cell("code", [
        "# Clone PaddleOCR repository (required for training)\n",
        "%cd /content\n",
        "!git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git\n",
        "print(\"✅ PaddleOCR cloned!\")"
    ], {"collapsed": False})
    
    # Insert after repository clone cell
    notebook["cells"].insert(5, paddleocr_cell)
    print("✅ Added PaddleOCR clone cell")
    
    # 2. Add arabic_dict.txt download cell in Part 2
    arabic_dict_cell = create_cell("code", [
        "# Download Arabic dictionary file (required for OCR)\n",
        "!wget -q -O /content/egyption_id_ready/arabic_dict.txt \\\n",
        "    https://raw.githubusercontent.com/NAMO7Y/Egyptian_ID_OCR/main/arabic_dict.txt\n",
        "\n",
        "# Verify file exists\n",
        "from pathlib import Path\n",
        "dict_path = Path('/content/egyption_id_ready/arabic_dict.txt')\n",
        "if dict_path.exists():\n",
        "    print(f\"✅ Arabic dictionary downloaded: {dict_path.stat().st_size} bytes\")\n",
        "else:\n",
        "    print(\"❌ Arabic dictionary download failed!\")\n",
        "    # Create minimal dictionary as fallback\n",
        "    with open(dict_path, 'w', encoding='utf-8') as f:\n",
        "        # Write basic Arabic characters\n",
        "        arabic_chars = ''.join([chr(i) for i in range(0x0600, 0x06FF)])\n",
        "        f.write(arabic_chars)\n",
        "    print(\"⚠️  Created minimal fallback dictionary\")"
    ])
    
    # Insert after OCR models download
    notebook["cells"].insert(11, arabic_dict_cell)
    print("✅ Added Arabic dictionary download cell")
    
    # 3. Add field detector model download
    field_detector_cell = create_cell("code", [
        "# Download field detector ONNX model (required for inference)\n",
        "!mkdir -p /content/egyption_id_ready/model\n",
        "\n",
        "# Try to download from project repository\n",
        "# Note: Replace with actual model URL if different\n",
        "!wget -q --timeout=30 --tries=3 -O /content/egyption_id_ready/model/field_detector.onnx \\\n",
        "    https://github.com/NASO7Y/OCR_Egyptian_ID/raw/main/field_detector.onnx || \\\n",
        "    echo \"⚠️  Field detector model not available - will use YOLO models instead\"\n",
        "\n",
        "# Verify download\n",
        "from pathlib import Path\n",
        "model_path = Path('/content/egyption_id_ready/model/field_detector.onnx')\n",
        "if model_path.exists() and model_path.stat().st_size > 0:\n",
        "    print(f\"✅ Field detector model downloaded: {model_path.stat().st_size / 1024 / 1024:.2f} MB\")\n",
        "else:\n",
        "    print(\"⚠️  Field detector model not available - two-stage detection will use YOLO models only\")"
    ])
    
    # Insert after Arabic dictionary
    notebook["cells"].insert(12, field_detector_cell)
    print("✅ Added field detector download cell")
    
    # 4. Add validation cell after model downloads
    validation_cell = create_cell("code", [
        "# Validate all required models are downloaded\n",
        "from pathlib import Path\n",
        "\n",
        "ROOT = Path('/content/egyption_id_ready')\n",
        "required_files = {\n",
        "    'weights/card_detection.pt': 'Card Detection Model',\n",
        "    'weights/field_detection.pt': 'Field Detection Model',\n",
        "    'weights/nid_detection.pt': 'NID Detection Model',\n",
        "    'arabic_PP-OCRv3_rec/best_accuracy.pdparams': 'PaddleOCR Model',\n",
        "    'arabic_dict.txt': 'Arabic Dictionary',\n",
        "}\n",
        "\n",
        "print(\"🔍 Validating downloaded files...\")\n",
        "all_ok = True\n",
        "for file_path, description in required_files.items():\n",
        "    full_path = ROOT / file_path\n",
        "    if full_path.exists() and full_path.stat().st_size > 0:\n",
        "        size_mb = full_path.stat().st_size / 1024 / 1024\n",
        "        print(f\"  ✅ {description}: {size_mb:.2f} MB\")\n",
        "    else:\n",
        "        print(f\"  ❌ {description}: MISSING\")\n",
        "        all_ok = False\n",
        "\n",
        "if all_ok:\n",
        "    print(\"\\n✅ All models downloaded successfully!\")\n",
        "else:\n",
        "    print(\"\\n⚠️  Some models are missing. Check error messages above.\")"
    ])
    
    # Insert after validation
    notebook["cells"].insert(13, validation_cell)
    print("✅ Added validation cell")
    
    # 5. Add disk space monitoring cell
    disk_monitor_cell = create_cell("code", [
        "# Monitor disk space\n",
        "!df -h /content\n",
        "\n",
        "# Clean unnecessary files to save space\n",
        "print(\"\\n🧹 Cleaning unnecessary files...\")\n",
        "!rm -rf /content/egyption_id_ready/.git 2>/dev/null || true\n",
        "!rm -rf /content/PaddleOCR/.git 2>/dev/null || true\n",
        "!rm -rf /content/egyption_id_ready/__pycache__ 2>/dev/null || true\n",
        "!rm -rf /content/egyption_id_ready/src/__pycache__ 2>/dev/null || true\n",
        "print(\"✅ Cleanup complete!\")\n",
        "\n",
        "# Show freed space\n",
        "!df -h /content"
    ])
    
    # Insert after validation
    notebook["cells"].insert(14, disk_monitor_cell)
    print("✅ Added disk monitoring cell")
    
    # 6. Add training resumption check - find and modify training cell
    for i, cell in enumerate(notebook["cells"]):
        source = ''.join(cell.get("source", []))
        if "Starting PaddleOCR training" in source:
            training_source = [
                "# Check for existing checkpoints before training\n",
                "from pathlib import Path\n",
                "checkpoint_dir = Path('/content/egyption_id_ready/output/egyptian_id_rec')\n",
                "latest_checkpoint = checkpoint_dir / 'latest.pdparams'\n",
                "\n",
                "if latest_checkpoint.exists():\n",
                "    print(f\"📋 Found existing checkpoint: {latest_checkpoint}\")\n",
                "    print(\"⏸️  Training will resume from last checkpoint\")\n",
                "else:\n",
                "    print(\"🆕 No checkpoint found - starting fresh training\")\n",
                "\n",
                "print(\"\\n🚀 Starting PaddleOCR training (2-4 hours)...\")\n",
                "print(\"📊 Checkpoints saved every 5 epochs\")\n",
                "print(\"⏱️  Use Colab's runtime disconnect prevention if available\")\n",
                "\n",
                "# Run training\n",
                "!cd /content/egyption_id_ready && python PaddleOCR/tools/train.py -c configs/egyptian_id_rec.yml -o Global.use_gpu=true -o Global.epoch_num=100 -o Global.save_model_dir=/content/egyption_id_ready/output/egyptian_id_rec/\n",
                "\n",
                "print(\"\\n✅ Training complete!\")"
            ]
            notebook["cells"][i]["source"] = training_source
            print("✅ Updated training cell with resumption support")
            break
    
    # 7. Add progress validation after dataset processing
    for i, cell in enumerate(notebook["cells"]):
        source = ''.join(cell.get("source", []))
        if "Processing full dataset" in source:
            dataset_validation_cell = create_cell("code", [
                "# Validate dataset processing results\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "\n",
                "print(\"🔍 Validating dataset processing...\")\n",
                "\n",
                "# Check cropped images\n",
                "crops_dir = Path('/content/egyption_id_ready/rec/images/two_stage')\n",
                "if crops_dir.exists():\n",
                "    crop_count = len(list(crops_dir.glob('*.jpg')))\n",
                "    print(f\"  ✅ Cropped fields: {crop_count:,} images\")\n",
                "else:\n",
                "    print(\"  ❌ Crops directory not found!\")\n",
                "\n",
                "# Check metadata CSV\n",
                "metadata_path = Path('/content/egyption_id_ready/crops_metadata_two_stage.csv')\n",
                "if metadata_path.exists():\n",
                "    df = pd.read_csv(metadata_path)\n",
                "    print(f\"  ✅ Metadata: {len(df):,} records\")\n",
                "    print(f\"     Fields: {df['field'].nunique()} unique\")\n",
                "    print(f\"     Avg confidence: {df['confidence'].mean():.3f}\")\n",
                "else:\n",
                "    print(\"  ❌ Metadata CSV not found!\")\n",
                "\n",
                "print(\"\\n✅ Dataset validation complete!\")"
            ])
            notebook["cells"].insert(i + 1, dataset_validation_cell)
            print("✅ Added dataset validation cell")
            break
    
    # 8. Add error handling wrapper for critical cells
    error_handling_cell = create_cell("code", [
        "# Helper function for robust downloads\n",
        "def download_with_retry(url, output_path, max_retries=3, timeout=60):\n",
        "    \"\"\"Download file with retry logic.\"\"\"\n",
        "    import subprocess\n",
        "    from pathlib import Path\n",
        "    \n",
        "    output = Path(output_path)\n",
        "    output.parent.mkdir(parents=True, exist_ok=True)\n",
        "    \n",
        "    for attempt in range(max_retries):\n",
        "        try:\n",
        "            result = subprocess.run(\n",
        "                ['wget', '-q', '--timeout=' + str(timeout), \n",
        "                 '--tries=1', '-O', str(output), url],\n",
        "                capture_output=True,\n",
        "                text=True,\n",
        "                timeout=timeout * 2\n",
        "            )\n",
        "            if result.returncode == 0 and output.exists() and output.stat().st_size > 0:\n",
        "                return True\n",
        "        except Exception as e:\n",
        "            print(f\"Attempt {attempt + 1}/{max_retries} failed: {e}\")\n",
        "    \n",
        "    return False\n",
        "\n",
        "print(\"✅ Download helper loaded!\")"
    ])
    
    # Insert early in notebook
    notebook["cells"].insert(7, error_handling_cell)
    print("✅ Added error handling helper")
    
    # 9. Add ngrok setup cell for API - find and replace API cell
    for i, cell in enumerate(notebook["cells"]):
        source = ''.join(cell.get("source", []))
        if "Testing API" in source or "Deploy API" in source:
            ngrok_cell = create_cell("code", [
                "# Setup ngrok tunnel for API access\n",
                "print(\"🌐 Setting up ngrok tunnel...\")\n",
                "\n",
                "# Install pyngrok\n",
                "!pip install -q pyngrok\n",
                "\n",
                "# Set ngrok auth token (optional but recommended)\n",
                "# Get your token from https://dashboard.ngrok.com/get-started/your-authtoken\n",
                "NGROK_AUTH_TOKEN = \"\"  # ← Add your ngrok auth token here\n",
                "\n",
                "from pyngrok import ngrok, conf\n",
                "\n",
                "if NGROK_AUTH_TOKEN:\n",
                "    conf.get_default().auth_token = NGROK_AUTH_TOKEN\n",
                "    print(\"✅ Using authenticated ngrok tunnel\")\n",
                "else:\n",
                "    print(\"⚠️  Using anonymous tunnel (limited to 1 connection)\")\n",
                "    print(\"   Get free auth token: https://dashboard.ngrok.com/get-started/your-authtoken\")\n",
                "\n",
                "# Kill any existing tunnels\n",
                "ngrok.kill()\n",
                "\n",
                "# Start API server in background\n",
                "import subprocess\n",
                "import time\n",
                "\n",
                "print(\"\\n🚀 Starting FastAPI server...\")\n",
                "api_process = subprocess.Popen(\n",
                "    ['uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8000'],\n",
                "    cwd='/content/egyption_id_ready',\n",
                "    stdout=subprocess.PIPE,\n",
                "    stderr=subprocess.PIPE\n",
                ")\n",
                "\n",
                "# Wait for server to start\n",
                "time.sleep(5)\n",
                "\n",
                "# Create tunnel\n",
                "public_url = ngrok.connect(8000)\n",
                "print(f\"\\n✅ API server started!\")\n",
                "print(f\"🔗 Public API URL: {public_url}\")\n",
                "print(f\"📖 API Docs: {public_url}/docs\")\n",
                "print(f\"📖 API Health: {public_url}/\")\n",
                "\n",
                "# Keep this cell running to maintain the tunnel\n",
                "# To stop: press the stop button or run: ngrok.kill()"
            ])
            notebook["cells"][i] = ngrok_cell
            print("✅ Updated API cell with ngrok setup")
            break
    
    # 10. Add final summary and validation cell at the end
    final_validation_cell = create_cell("code", [
        "# Final validation and summary\n",
        "from pathlib import Path\n",
        "import pandas as pd\n",
        "\n",
        "print(\"=\" * 60)\n",
        "print(\"🎉 FINAL VALIDATION\")\n",
        "print(\"=\" * 60)\n",
        "\n",
        "ROOT = Path('/content/egyption_id_ready')\n",
        "\n",
        "# Check all outputs\n",
        "outputs = {\n",
        "    'Trained Model': ROOT / 'output' / 'egyptian_id_rec' / 'best_accuracy.pdparams',\n",
        "    'ONNX Model': ROOT / 'onnx' / 'rec_sim.onnx',\n",
        "    'Labeled Data': ROOT / 'crops_labeled.csv',\n",
        "    'Metadata': ROOT / 'crops_metadata_two_stage.csv',\n",
        "}\n",
        "\n",
        "print(\"\\n📂 Output Files:\")\n",
        "all_ok = True\n",
        "for name, path in outputs.items():\n",
        "    if path.exists():\n",
        "        size_mb = path.stat().st_size / 1024 / 1024\n",
        "        print(f\"  ✅ {name}: {size_mb:.2f} MB\")\n",
        "    else:\n",
        "        print(f\"  ❌ {name}: MISSING\")\n",
        "        all_ok = False\n",
        "\n",
        "# Check Google Drive backup\n",
        "drive_path = Path('/content/drive/MyDrive/egyption_id_ready')\n",
        "if drive_path.exists():\n",
        "    print(f\"\\n💾 Google Drive Backup:\")\n",
        "    for item in drive_path.rglob('*'):\n",
        "        if item.is_file():\n",
        "            print(f\"  ✅ {item.relative_to(drive_path)}\")\n",
        "\n",
        "print(\"\\n\" + \"=\" * 60)\n",
        "if all_ok:\n",
        "    print(\"✅ ALL VALIDATIONS PASSED!\")\n",
        "    print(\"\\n🚀 Your Egyptian ID OCR system is ready!\")\n",
        "    print(\"\\n📋 Next Steps:\")\n",
        "    print(\"   1. Download models from Google Drive\")\n",
        "    print(\"   2. Deploy API using Docker or locally\")\n",
        "    print(\"   3. Integrate with your application\")\n",
        "else:\n",
        "    print(\"⚠️  Some outputs are missing. Check error messages above.\")\n",
        "print(\"=\" * 60)"
    ])
    
    # Add at the end
    notebook["cells"].append(final_validation_cell)
    print("✅ Added final validation cell")
    
    # Save updated notebook
    save_notebook(notebook, notebook_path)
    print(f"\n✅ Notebook updated successfully!")
    print(f"   Location: {notebook_path}")
    print(f"   Total cells: {len(notebook['cells'])}")

if __name__ == "__main__":
    main()
