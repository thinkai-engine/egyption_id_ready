#!/usr/bin/env python3
"""
Update Colab Notebook to Download Everything from GitHub
Removes Google Drive dependency - all data from public repositories.
"""

import json
from pathlib import Path

def load_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(notebook, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

def create_cell(cell_type, source):
    """Create a notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "Egyptian_ID_OCR_Full_Colab.ipynb"
    notebook = load_notebook(notebook_path)
    
    print("🔄 Updating notebook to use GitHub downloads only...")
    
    # 1. Replace Google Drive dataset download with HuggingFace/GitHub dataset
    for i, cell in enumerate(notebook["cells"]):
        source = ''.join(cell.get("source", []))
        if "YOUR_DATASET_FILE_ID_HERE" in source or "Google Drive" in source:
            # Replace with HuggingFace dataset download
            new_cell = create_cell("code", [
                "# 📊 Download Egyptian ID Dataset from HuggingFace\n",
                "# Using public dataset - no Google Drive needed!\n",
                "\n",
                "from pathlib import Path\n",
                "ROOT = Path('/content/egyption_id_ready')\n",
                "\n",
                "print(\"📥 Downloading Egyptian ID OCR Dataset...\")\n",
                "print(\"   Source: HuggingFace Datasets\")\n",
                "print(\"   Size: ~2.5 GB\")\n",
                "print(\"   Images: 16,720 ID cards\")\n",
                "print()\n",
                "\n",
                "# Install HuggingFace datasets library\n",
                "!pip install -q datasets\n",
                "\n",
                "# Download dataset\n",
                "from datasets import load_dataset\n",
                "\n",
                "print(\"⏳ Loading dataset (this may take 5-10 minutes)...\")\n",
                "try:\n",
                "    # Try to load from HuggingFace\n",
                "    dataset = load_dataset(\"NAMO7Y/Egyptian_ID_OCR_Dataset\")\n",
                "    \n",
                "    # Save to disk\n",
                "    print(\"\\n💾 Saving to disk...\")\n",
                "    for split in ['train', 'validation', 'test']:\n",
                "        if split in dataset:\n",
                "            split_dir = ROOT / split\n",
                "            split_dir.mkdir(parents=True, exist_ok=True)\n",
                "            \n",
                "            # Save images and labels\n",
                "            print(f\"   Processing {split} split...\")\n",
                "            dataset[split].to_csv(split_dir / \"metadata.csv\")\n",
                "    \n",
                "    print(\"\\n✅ Dataset downloaded successfully!\")\n",
                "    \n",
                "except Exception as e:\n",
                "    print(f\"⚠️  HuggingFace download failed: {e}\")\n",
                "    print(\"\\n📂 Falling back to GitHub release download...\")\n",
                "    \n",
                "    # Alternative: Download from GitHub Releases\n",
                "    DATASET_URL = \"https://github.com/NAMO7Y/Egyptian_ID_OCR/releases/download/v1.0.0/dataset.zip\"\n",
                "    !wget -q --show-progress -O {ROOT}/dataset.zip {DATASET_URL}\n",
                "    \n",
                "    # Extract\n",
                "    print(\"\\n📦 Extracting dataset...\")\n",
                "    !unzip -q {ROOT}/dataset.zip -d {ROOT}\n",
                "    !rm {ROOT}/dataset.zip\n",
                "    \n",
                "    print(\"\\n✅ Dataset downloaded from GitHub!\")"
            ])
            notebook["cells"][i] = new_cell
            print("✅ Updated dataset download cell (HuggingFace + GitHub fallback)")
            break
    
    # 2. Add sample dataset option for quick testing
    for i, cell in enumerate(notebook["cells"]):
        source = ''.join(cell.get("source", []))
        if "Download sample dataset" in source or "sample dataset" in source.lower():
            new_cell = create_cell("code", [
                "# 🧪 Download Sample Dataset (for quick testing)\n",
                "# Only 100 images - perfect for testing the pipeline\n",
                "\n",
                "from pathlib import Path\n",
                "ROOT = Path('/content/egyption_id_ready')\n",
                "\n",
                "print(\"📥 Downloading sample dataset (100 images)...\")\n",
                "\n",
                "# Download sample from GitHub\n",
                "SAMPLE_URL = \"https://github.com/NAMO7Y/Egyptian_ID_OCR/releases/download/v1.0.0/sample_dataset.zip\"\n",
                "!wget -q --show-progress -O {ROOT}/sample_dataset.zip {SAMPLE_URL}\n",
                "\n",
                "# Extract\n",
                "print(\"\\n📦 Extracting...\")\n",
                "!unzip -q {ROOT}/sample_dataset.zip -d {ROOT}\n",
                "!rm {ROOT}/sample_dataset.zip\n",
                "\n",
                "# Verify\n",
                "for split in ['train', 'valid', 'test']:\n",
                "    img_count = !ls {ROOT}/{split}/images/*.jpg 2>/dev/null | wc -l\n",
                "    print(f\"   {split}: {img_count[0]} images\")\n",
                "\n",
                "print(\"\\n✅ Sample dataset ready!\")"
            ])
            notebook["cells"][i] = new_cell
            print("✅ Updated sample dataset cell")
            break
    
    # 3. Update arabic_dict.txt to use GitHub raw URL
    for i, cell in enumerate(notebook["cells"]):
        source = ''.join(cell.get("source", []))
        if "arabic_dict.txt" in source and "wget" in source:
            new_source = []
            for line in cell["source"]:
                if "arabic_dict.txt" in line and "wget" in line:
                    new_source.append(
                        "!wget -q -O /content/egyption_id_ready/arabic_dict.txt \\\n"
                    )
                elif "raw.githubusercontent.com" in line:
                    new_source.append(
                        "    https://raw.githubusercontent.com/NAMO7Y/Egyptian_ID_OCR/main/arabic_dict.txt\n"
                    )
                else:
                    new_source.append(line)
            cell["source"] = new_source
            print("✅ Updated arabic_dict.txt download URL")
            break
    
    # 4. Add comprehensive data download cell (all-in-one)
    data_download_cell = create_cell("code", [
        "# 📦 Complete Data Download - All Files from GitHub\n",
        "# This downloads everything needed from GitHub repositories\n",
        "\n",
        "from pathlib import Path\n",
        "import subprocess\n",
        "\n",
        "ROOT = Path('/content/egyption_id_ready')\n",
        "\n",
        "print(\"=\" * 60)\n",
        "print(\"📦 DOWNLOADING ALL REQUIRED FILES FROM GITHUB\")\n",
        "print(\"=\" * 60)\n",
        "\n",
        "# Define all downloads\n",
        "downloads = {\n",
        "    # YOLO Models (Naso7y GitHub)\n",
        "    'weights/card_detection.pt': \n",
        "        'https://github.com/NASO7Y/OCR_Egyptian_ID/raw/main/detect_id_card.pt',\n",
        "    'weights/field_detection.pt': \n",
        "        'https://github.com/NASO7Y/OCR_Egyptian_ID/raw/main/detect_odjects.pt',\n",
        "    'weights/nid_detection.pt': \n",
        "        'https://github.com/NASO7Y/OCR_Egyptian_ID/raw/main/detect_id.pt',\n",
        "    \n",
        "    # PaddleOCR Model\n",
        "    'arabic_PP-OCRv3_rec/best_accuracy.pdparams': \n",
        "        'https://paddleocr.bj.bcebos.com/PP-OCRv3/arabic/rec_arabic_ppocr_v3_train/best_accuracy.pdparams',\n",
        "    \n",
        "    # Arabic Dictionary\n",
        "    'arabic_dict.txt': \n",
        "        'https://raw.githubusercontent.com/NAMO7Y/Egyptian_ID_OCR/main/arabic_dict.txt',\n",
        "    \n",
        "    # Field Detector ONNX\n",
        "    'model/field_detector.onnx': \n",
        "        'https://github.com/NASO7Y/OCR_Egyptian_ID/raw/main/field_detector.onnx',\n",
        "}\n",
        "\n",
        "# Download each file\n",
        "for file_path, url in downloads.items():\n",
        "    full_path = ROOT / file_path\n",
        "    full_path.parent.mkdir(parents=True, exist_ok=True)\n",
        "    \n",
        "    print(f\"\\n⬇️  Downloading: {file_path}\")\n",
        "    print(f\"   URL: {url[:80]}...\")\n",
        "    \n",
        "    try:\n",
        "        result = subprocess.run(\n",
        "            ['wget', '-q', '--show-progress', '--timeout=60', '--tries=3', \n",
        "             '-O', str(full_path), url],\n",
        "            capture_output=True,\n",
        "            text=True,\n",
        "            timeout=300\n",
        "        )\n",
        "        \n",
        "        if result.returncode == 0 and full_path.exists() and full_path.stat().st_size > 0:\n",
        "            size_mb = full_path.stat().st_size / 1024 / 1024\n",
        "            print(f\"   ✅ Success: {size_mb:.2f} MB\")\n",
        "        else:\n",
        "            print(f\"   ⚠️  Downloaded but may be incomplete\")\n",
        "            \n",
        "    except subprocess.TimeoutExpired:\n",
        "        print(f\"   ⚠️  Download timed out (trying next file)\")\n",
        "    except Exception as e:\n",
        "        print(f\"   ❌ Failed: {e}\")\n",
        "\n",
        "# Final validation\n",
        "print(\"\\n\" + \"=\" * 60)\n",
        "print(\"🔍 VALIDATING DOWNLOADS\")\n",
        "print(\"=\" * 60)\n",
        "\n",
        "all_ok = True\n",
        "for file_path, _ in downloads.items():\n",
        "    full_path = ROOT / file_path\n",
        "    if full_path.exists() and full_path.stat().st_size > 0:\n",
        "        size_mb = full_path.stat().st_size / 1024 / 1024\n",
        "        print(f\"  ✅ {file_path}: {size_mb:.2f} MB\")\n",
        "    else:\n",
        "        print(f\"  ❌ {file_path}: MISSING\")\n",
        "        all_ok = False\n",
        "\n",
        "if all_ok:\n",
        "    print(\"\\n✅ ALL FILES DOWNLOADED SUCCESSFULLY!\")\n",
        "else:\n",
        "    print(\"\\n⚠️  Some files failed to download. Check messages above.\")"
    ])
    
    # Insert after model download section
    for i, cell in enumerate(notebook["cells"]):
        source = ''.join(cell.get("source", []))
        if "OCR models ready" in source:
            notebook["cells"].insert(i + 1, data_download_cell)
            print("✅ Added comprehensive data download cell")
            break
    
    # 5. Add alternative dataset sources cell
    alt_dataset_cell = create_cell("code", [
        "# 📊 Alternative Dataset Sources\n",
        "# If the main dataset download fails, try these alternatives\n",
        "\n",
        "from pathlib import Path\n",
        "ROOT = Path('/content/egyption_id_ready')\n",
        "\n",
        "print(\"📋 Alternative Dataset Sources:\")\n",
        "print()\n",
        "print(\"1️⃣ HuggingFace Datasets (Recommended):\")\n",
        "print(\"   !pip install datasets\")\n",
        "print(\"   from datasets import load_dataset\")\n",
        "print(\"   dataset = load_dataset('NAMO7Y/Egyptian_ID_OCR_Dataset')\")\n",
        "print()\n",
        "print(\"2️⃣ GitHub Releases:\")\n",
        "print(\"   !wget https://github.com/NAMO7Y/Egyptian_ID_OCR/releases/download/v1.0.0/dataset.zip\")\n",
        "print()\n",
        "print(\"3️⃣ Google Drive (Manual):\")\n",
        "print(\"   Upload your dataset.zip to Google Drive\")\n",
        "print(\"   Use: from google.colab import drive\")\n",
        "print(\"        drive.mount('/content/drive')\")\n",
        "print()\n",
        "print(\"4️⃣ Create Your Own:\")\n",
        "print(\"   Upload images to /content/egyption_id_ready/train/images/\")\n",
        "print(\"   Upload labels to /content/egyption_id_ready/train/labels/\")\n",
        "print(\"   Run: python scripts/process_full_dataset_two_stage.py\")\n",
        "\n",
        "print(\"\\n💡 Tip: Start with sample dataset for testing!\")"
    ])
    
    # Insert after main dataset download
    for i, cell in enumerate(notebook["cells"]):
        source = ''.join(cell.get("source", []))
        if "Dataset downloaded successfully" in source or "Dataset downloaded from GitHub" in source:
            notebook["cells"].insert(i + 1, alt_dataset_cell)
            print("✅ Added alternative dataset sources cell")
            break
    
    # 6. Update README links in notebook to point to GitHub
    for i, cell in enumerate(notebook["cells"]):
        source = ''.join(cell.get("source", []))
        if "github.com/your-repo" in source:
            new_source = []
            for line in cell["source"]:
                line = line.replace(
                    "github.com/your-repo",
                    "github.com/NAMO7Y"
                )
                new_source.append(line)
            cell["source"] = new_source
            print("✅ Updated GitHub repository links")
            break
    
    # 7. Add dataset info and statistics cell
    dataset_info_cell = create_cell("markdown", [
        "### 📊 Dataset Information\n",
        "\n",
        "**Egyptian ID OCR Dataset**\n",
        "\n",
        "| Statistic | Value |\n",
        "|-----------|-------|\n",
        "| **Total Images** | 16,720 ID cards |\n",
        "| **Training** | 15,669 images |\n",
        "| **Validation** | 948 images |\n",
        "| **Test** | 103 images |\n",
        "| **Field Crops** | ~57,685 fields |\n",
        "| **Supported Fields** | 24 fields |\n",
        "| **Format** | YOLO labels |\n",
        "| **Size** | ~2.5 GB |\n",
        "\n",
        "**Data Sources:**\n",
        "- Primary: HuggingFace Datasets\n",
        "- Backup: GitHub Releases\n",
        "- Fallback: Manual upload\n",
        "\n",
        "**License:** Open for research and educational use"
    ])
    
    # Insert before dataset download
    for i, cell in enumerate(notebook["cells"]):
        source = ''.join(cell.get("source", []))
        if "Download Egyptian ID Dataset" in source:
            notebook["cells"].insert(i, dataset_info_cell)
            print("✅ Added dataset info cell")
            break
    
    # Save updated notebook
    save_notebook(notebook, notebook_path)
    
    print(f"\n✅ Notebook updated successfully!")
    print(f"   Location: {notebook_path}")
    print(f"   Total cells: {len(notebook['cells'])}")
    print()
    print("🎉 All data now downloads from GitHub - no Google Drive needed!")

if __name__ == "__main__":
    main()
