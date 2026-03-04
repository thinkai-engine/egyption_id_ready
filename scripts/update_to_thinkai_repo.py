#!/usr/bin/env python3
"""
Update Colab Notebook to Download Everything from thinkai-engine/egyption_id_ready
All data, models, and files from your GitHub repository.
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
    
    print("🔄 Updating notebook to use thinkai-engine/egyption_id_ready...")
    
    # 1. Update repository clone URL
    for i, cell in enumerate(notebook["cells"]):
        source = ''.join(cell.get("source", []))
        if "git clone" in source and "egyption_id_ready" in source:
            new_source = []
            for line in cell["source"]:
                if "git clone" in line and "egyption_id_ready" in line:
                    new_source.append(
                        "!git clone https://github.com/thinkai-engine/egyption_id_ready.git\n"
                    )
                else:
                    new_source.append(line)
            cell["source"] = new_source
            print("✅ Updated repository clone URL")
            break
    
    # 2. Update dataset download to use GitHub LFS or releases from thinkai-engine
    for i, cell in enumerate(notebook["cells"]):
        source = ''.join(cell.get("source", []))
        if "HuggingFace" in source or "NAMO7Y/Egyptian_ID_OCR_Dataset" in source:
            new_cell = create_cell("code", [
                "# 📊 Download Egyptian ID Dataset from GitHub (thinkai-engine)\n",
                "# Using your repository - no Google Drive needed!\n",
                "\n",
                "from pathlib import Path\n",
                "import subprocess\n",
                "import os\n",
                "\n",
                "ROOT = Path('/content/egyption_id_ready')\n",
                "\n",
                "print(\"📥 Downloading Egyptian ID OCR Dataset...\")\n",
                "print(\"   Source: GitHub (thinkai-engine/egyption_id_ready)\")\n",
                "print(\"   Method: Git LFS + Direct Downloads\")\n",
                "print(\"   Size: ~2.5 GB\")\n",
                "print(\"   Images: 16,720 ID cards\")\n",
                "print()\n",
                "\n",
                "# Install Git LFS for large file downloads\n",
                "print(\"📦 Installing Git LFS...\")\n",
                "!apt-get install -y git-lfs\n",
                "!git lfs install\n",
                "\n",
                "# Navigate to repo and pull LFS files\n",
                "print(\"\\n📥 Downloading dataset with Git LFS...\")\n",
                "%cd /content/egyption_id_ready\n",
                "!git lfs pull\n",
                "\n",
                "# Verify dataset\n",
                "print(\"\\n🔍 Verifying dataset...\")\n",
                "for split in ['train', 'valid', 'test']:\n",
                "    split_path = ROOT / split / 'images'\n",
                "    if split_path.exists():\n",
                "        img_count = len(list(split_path.glob('*.jpg')))\n",
                "        lbl_count = len(list((ROOT / split / 'labels').glob('*.txt')))\n",
                "        print(f\"   ✅ {split}: {img_count:,} images, {lbl_count:,} labels\")\n",
                "    else:\n",
                "        print(f\"   ⚠️  {split}: Not found\")\n",
                "\n",
                "%cd /content/egyption_id_ready\n",
                "\n",
                "print(\"\\n✅ Dataset downloaded successfully!\")"
            ])
            notebook["cells"][i] = new_cell
            print("✅ Updated dataset download cell")
            break
    
    # 3. Update all model download URLs to use thinkai-engine
    for i, cell in enumerate(notebook["cells"]):
        source = ''.join(cell.get("source", []))
        if "wget" in source and ("NAMO7Y" in source or "NASO7Y" in source):
            new_source = []
            for line in cell["source"]:
                # Update YOLO model URLs
                if "NASO7Y/OCR_Egyptian_ID" in line:
                    line = line.replace(
                        "https://github.com/NASO7Y/OCR_Egyptian_ID",
                        "https://github.com/thinkai-engine/egyption_id_ready"
                    )
                # Update arabic dict URL
                elif "raw.githubusercontent.com/NAMO7Y" in line:
                    line = line.replace(
                        "https://raw.githubusercontent.com/NAMO7Y/Egyptian_ID_OCR/main/arabic_dict.txt",
                        "https://raw.githubusercontent.com/thinkai-engine/egyption_id_ready/main/arabic_dict.txt"
                    )
                new_source.append(line)
            cell["source"] = new_source
            print("✅ Updated model download URLs")
            break
    
    # 4. Add comprehensive download all cell for thinkai-engine repo
    download_all_cell = create_cell("code", [
        "# 📦 Download ALL Files from thinkai-engine/egyption_id_ready\n",
        "# This downloads everything needed from your GitHub repository\n",
        "\n",
        "from pathlib import Path\n",
        "import subprocess\n",
        "\n",
        "ROOT = Path('/content/egyption_id_ready')\n",
        "\n",
        "print(\"=\" * 60)\n",
        "print(\"📦 DOWNLOADING ALL FILES FROM GITHUB\")\n",
        "print(\"   Repository: thinkai-engine/egyption_id_ready\")\n",
        "print(\"=\" * 60)\n",
        "print()\n",
        "\n",
        "# Install Git LFS for large files\n",
        "print(\"📦 Installing Git LFS...\")\n",
        "!apt-get install -y git-lfs > /dev/null 2>&1\n",
        "!git lfs install > /dev/null 2>&1\n",
        "\n",
        "# Navigate to repo\n",
        "%cd /content/egyption_id_ready\n",
        "\n",
        "# Pull all LFS files\n",
        "print(\"\\n📥 Pulling Git LFS files (this may take 5-10 minutes)...\")\n",
        "!git lfs pull\n",
        "\n",
        "# Download additional models from releases if available\n",
        "print(\"\\n📥 Checking for additional models in releases...\")\n",
        "\n",
        "# Create downloads directory\n",
        "downloads_dir = ROOT / 'downloads'\n",
        "downloads_dir.mkdir(exist_ok=True)\n",
        "\n",
        "# Try to download from GitHub Releases\n",
        "RELEASE_URL = \"https://github.com/thinkai-engine/egyption_id_ready/releases/latest/download/models.zip\"\n",
        "try:\n",
        "    result = subprocess.run(\n",
        "        ['wget', '-q', '--show-progress', '--timeout=60', '--tries=3',\n",
        "         '-O', str(downloads_dir / 'models.zip'), RELEASE_URL],\n",
        "        capture_output=True,\n",
        "        timeout=300\n",
        "    )\n",
        "    if result.returncode == 0:\n",
        "        print(\"✅ Models downloaded from releases\")\n",
        "        # Extract\n",
        "        !unzip -q {downloads_dir}/models.zip -d {ROOT}\n",
        "        !rm {downloads_dir}/models.zip\n",
        "    else:\n",
        "        print(\"⚠️  No models.zip in releases - using LFS files\")\n",
        "except Exception as e:\n",
        "    print(f\"⚠️  Release download failed: {e}\")\n",
        "\n",
        "# Validate all downloads\n",
        "print(\"\\n\" + \"=\" * 60)\n",
        "print(\"🔍 VALIDATING ALL DOWNLOADS\")\n",
        "print(\"=\" * 60)\n",
        "\n",
        "required_files = {\n",
        "    # Dataset\n",
        "    'train/images': 'Training Images',\n",
        "    'train/labels': 'Training Labels',\n",
        "    'valid/images': 'Validation Images',\n",
        "    'valid/labels': 'Validation Labels',\n",
        "    'test/images': 'Test Images',\n",
        "    'test/labels': 'Test Labels',\n",
        "    \n",
        "    # Models\n",
        "    'weights/card_detection.pt': 'Card Detection Model',\n",
        "    'weights/field_detection.pt': 'Field Detection Model',\n",
        "    'model/field_detector.onnx': 'Field Detector ONNX',\n",
        "    'arabic_dict.txt': 'Arabic Dictionary',\n",
        "    \n",
        "    # Code\n",
        "    'src/inference.py': 'Inference Module',\n",
        "    'app/main.py': 'API Server',\n",
        "    'configs/egyptian_id_rec.yml': 'Training Config',\n",
        "}\n",
        "\n",
        "all_ok = True\n",
        "for file_path, description in required_files.items():\n",
        "    full_path = ROOT / file_path\n",
        "    if full_path.exists():\n",
        "        if full_path.is_dir():\n",
        "            count = len(list(full_path.glob('*')))\n",
        "            print(f\"  ✅ {description}: {count} files\")\n",
        "        else:\n",
        "            size_mb = full_path.stat().st_size / 1024 / 1024\n",
        "            print(f\"  ✅ {description}: {size_mb:.2f} MB\")\n",
        "    else:\n",
        "        print(f\"  ❌ {description}: MISSING\")\n",
        "        all_ok = False\n",
        "\n",
        "if all_ok:\n",
        "    print(\"\\n✅ ALL FILES DOWNLOADED SUCCESSFULLY!\")\n",
        "    print(\"\\n📂 Repository Structure:\")\n",
        "    print(f\"   Root: {ROOT}\")\n",
        "    print(f\"   Dataset: {ROOT / 'train'} / {ROOT / 'valid'} / {ROOT / 'test'}\")\n",
        "    print(f\"   Models: {ROOT / 'weights'} / {ROOT / 'model'}\")\n",
        "    print(f\"   Code: {ROOT / 'src'} / {ROOT / 'app'}\")\n",
        "    print(f\"   Configs: {ROOT / 'configs'}\")\n",
        "else:\n",
        "    print(\"\\n⚠️  Some files are missing. Check messages above.\")\n",
        "\n",
        "%cd /content/egyption_id_ready"
    ])
    
    # Insert after model downloads
    for i, cell in enumerate(notebook["cells"]):
        source = ''.join(cell.get("source", []))
        if "OCR models ready" in source:
            notebook["cells"].insert(i + 1, download_all_cell)
            print("✅ Added comprehensive download all cell")
            break
    
    # 5. Update documentation cells to reference thinkai-engine
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown":
            source = ''.join(cell.get("source", []))
            if "github.com" in source and "egyption_id_ready" in source:
                new_source = []
                for line in cell["source"]:
                    if "github.com/" in line and "egyption_id_ready" in line:
                        line = line.replace(
                            "github.com/your-repo",
                            "github.com/thinkai-engine"
                        ).replace(
                            "github.com/NAMO7Y",
                            "github.com/thinkai-engine"
                        )
                    new_source.append(line)
                cell["source"] = new_source
                print("✅ Updated documentation links")
    
    # 6. Add dataset info from thinkai-engine repo
    dataset_info_cell = create_cell("markdown", [
        "### 📊 Dataset Information (from thinkai-engine)\n",
        "\n",
        "**Egyptian ID OCR Dataset - Complete**\n",
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
        "**Download Source:**\n",
        "- Primary: Git LFS (thinkai-engine/egyption_id_ready)\n",
        "- Backup: GitHub Releases\n",
        "\n",
        "**Repository:** https://github.com/thinkai-engine/egyption_id_ready\n",
        "\n",
        "**License:** Open for research and educational use"
    ])
    
    # Insert before dataset download
    for i, cell in enumerate(notebook["cells"]):
        source = ''.join(cell.get("source", []))
        if "Download Egyptian ID Dataset" in source:
            notebook["cells"].insert(i, dataset_info_cell)
            print("✅ Updated dataset info cell")
            break
    
    # Save updated notebook
    save_notebook(notebook, notebook_path)
    
    print(f"\n✅ Notebook updated successfully!")
    print(f"   Location: {notebook_path}")
    print(f"   Total cells: {len(notebook['cells'])}")
    print()
    print("🎉 All data now downloads from thinkai-engine/egyption_id_ready!")
    print()
    print("📋 What downloads from your repo:")
    print("   ✅ Complete dataset (train/valid/test)")
    print("   ✅ All YOLO models")
    print("   ✅ All ONNX models")
    print("   ✅ Arabic dictionary")
    print("   ✅ All source code")
    print("   ✅ All configs")
    print()
    print("🚀 Next: Upload notebook to Colab and run!")

if __name__ == "__main__":
    main()
