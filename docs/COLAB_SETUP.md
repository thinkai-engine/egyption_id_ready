# 🇪🇬 Egyptian ID OCR - Google Colab Setup Guide

This guide explains how to run the complete Egyptian ID OCR pipeline on Google Colab.

## 📓 Colab Notebook Location

**Notebook:** `notebooks/Egyptian_ID_OCR_Full_Colab.ipynb`

## 🚀 Quick Start

### Option 1: Upload to Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Upload notebook**
3. Upload `Egyptian_ID_OCR_Full_Colab.ipynb`
4. Click **Runtime** → **Change runtime type** → Select **GPU**
5. Run all cells sequentially

### Option 2: Open from GitHub

If your repository is on GitHub:

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **GitHub** tab
3. Enter your repository URL
4. Select `Egyptian_ID_OCR_Full_Colab.ipynb`
5. Click **Runtime** → **Change runtime type** → Select **GPU**

## ⚙️ Before Running - Required Setup

### 1. Update Dataset Link

In the notebook, find this cell:

```python
FILE_ID = "YOUR_DATASET_FILE_ID_HERE"
```

Replace with your actual Google Drive file ID.

**To get your Google Drive file ID:**

1. Upload your dataset ZIP to Google Drive
2. Right-click the file → **Share** → **Get link**
3. Change restriction to **Anyone with the link**
4. Copy the link: `https://drive.google.com/file/d/FILE_ID/view`
5. Extract the `FILE_ID` part

### 2. Update Repository URL

Find this cell:

```python
!git clone https://github.com/your-repo/egyption_id_ready.git
```

Replace with your actual repository URL.

## 📊 What the Notebook Does

### Part 1: Environment Setup (5 minutes)
- Check GPU availability
- Mount Google Drive (optional)
- Clone repository
- Install all dependencies
- Verify installation

### Part 2: Download Models (10 minutes)
- **YOLO Models** from GitHub NASO7Y:
  - `card_detection.pt` - Detects ID card
  - `field_detection.pt` - Detects fields on card
  - `nid_detection.pt` - Detects NID digits
- **OCR Models**:
  - EasyOCR (Arabic + English)
  - PaddleOCR pretrained model

### Part 3: Download Dataset (5-10 minutes)
- Download from Google Drive
- Extract and verify structure
- Display statistics

### Part 4: Build Dataset with Two-Stage Detection (5-10 minutes)
- Load two-stage YOLO detector
- Process all images through card → field detection
- Save cropped fields to `rec/images/two_stage/`
- Create metadata CSV

**Results:** ~57,685 field crops from 16,720 images

### Part 5: Label Crops with OCR (30-60 minutes)
- Extract text from all cropped fields
- Multiple OCR methods available:
  - **qari-airllm**: Qwen2-VL-2B (recommended for Colab)
  - **bakri-airllm**: Gemma-3-4B
  - **airllm**: Qwen2-VL-72B (slowest, most accurate)
  - **qari**: Standard QARI (requires more VRAM)
  - **bakri**: Standard Bakri
  - **gemini**: Google Gemini (requires API key)

### Part 6: Prepare Training Data (2 minutes)
- Convert labeled crops to PaddleOCR format
- Create `train.txt`, `valid.txt`, `test.txt`

### Part 7: Train PaddleOCR Model (2-4 hours)
- Fine-tune SVTR_LCNet on Egyptian ID data
- 100 epochs with checkpoints every 5 epochs
- Saves best accuracy model

### Part 8: Evaluate Model (5 minutes)
- Calculate CER (Character Error Rate)
- Calculate WER (Word Error Rate)
- Display per-field accuracy

### Part 9: Export to ONNX (5 minutes)
- Export PaddleOCR model to inference format
- Convert to ONNX
- Optimize with onnxsim
- **Result:** ~50-80% size reduction

### Part 10: Test Inference (2 minutes)
- Load complete OCR pipeline
- Test on sample images
- Display extracted fields

### Part 11: Save to Google Drive (2 minutes)
- Copy trained models to Drive
- Copy ONNX models
- Copy labeled data
- All outputs preserved after Colab session ends

## 📈 Expected Results

### Dataset Processing
| Split | Images | Success Rate | Field Crops |
|-------|--------|--------------|-------------|
| Train | 15,669 | 72.9% | 54,362 |
| Valid | 948 | 70.7% | 3,131 |
| Test | 103 | 37.9% | 192 |
| **Total** | **16,720** | **72.5%** | **57,685** |

### Training Metrics
- **Character Error Rate (CER):** < 5%
- **Word Error Rate (WER):** < 8%
- **Training Speed:** ~1000 samples/sec (GPU)

### Model Sizes
- **PaddleOCR (trained):** ~50 MB
- **ONNX (optimized):** ~30 MB

## ⚠️ Important Notes

### Colab Limitations
- **Session Timeout:** Colab free tier disconnects after 12 hours
- **GPU Availability:** T4 GPU typically available (15 GB VRAM)
- **Storage:** ~80 GB available in `/content/`
- **Files Lost:** All files in `/content/` deleted after session ends

### Solutions
1. **Save to Google Drive:** Use Part 11 to save all outputs
2. **Long Training:** Training may exceed session limit - save checkpoints
3. **Resume Training:** Modify config to resume from last checkpoint

### VRAM Requirements

| OCR Method | Minimum VRAM | Speed | Accuracy |
|------------|--------------|-------|----------|
| qari-airllm | 4 GB | Slow | High |
| bakri-airllm | 4 GB | Slow | High |
| airllm (72B) | 4 GB | Very Slow | Very High |
| qari | 8 GB | Fast | High |
| bakri | 8 GB | Fast | High |
| gemini | N/A (API) | Fast | Very High |

**Recommended for Colab Free Tier:** `qari-airllm` or `bakri-airllm`

## 🔧 Troubleshooting

### Issue: Download Failed
```
ERROR: cannot verify github.com's certificate
```
**Solution:** Use `--no-check-certificate` with wget:
```python
!wget --no-check-certificate -q --show-progress -O file.pt URL
```

### Issue: Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions:**
1. Reduce batch size in config: `Train.loader.batch_size_per_card: 64`
2. Use AirLLM methods with 4-bit quantization
3. Restart Colab runtime to clear memory

### Issue: Training Too Slow
**Solutions:**
1. Ensure GPU is selected: Runtime → Change runtime type → GPU
2. Check GPU usage: `!nvidia-smi`
3. Reduce number of epochs for testing

### Issue: Dataset Not Found
```
FileNotFoundError: [Errno 2] No such file or directory
```
**Solutions:**
1. Verify Google Drive file ID is correct
2. Ensure file sharing is set to "Anyone with the link"
3. Check file is a ZIP with correct structure

## 📦 Dataset Structure

Your dataset ZIP should contain:

```
dataset.zip
├── train/
│   ├── images/
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   └── labels/
│       ├── 001.txt
│       ├── 002.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Label Format (YOLO)
Each `.txt` file contains:
```
class_id x_center y_center width height
```

Example:
```
5 0.4523 0.6234 0.1234 0.0456  # name field
7 0.5123 0.7234 0.0834 0.0356  # national_id
```

## 🎯 Optimization Tips

### Faster Processing
1. **Limit dataset size** for testing:
   ```python
   !python scripts/process_full_dataset_two_stage.py --splits train --limit 100
   ```

2. **Use fewer epochs** for initial training:
   ```python
   -o Global.epoch_num=10  # Test with 10 epochs first
   ```

3. **Skip ONNX export** if not needed for deployment

### Better Accuracy
1. **Use higher accuracy OCR** for labeling:
   ```python
   OCR_METHOD = "airllm"  # Qwen2-VL-72B
   ```

2. **Train longer:**
   ```python
   -o Global.epoch_num=200
   ```

3. **Data augmentation:** Enable in config file

### Cost Optimization (Colab Pro)
- **Colab Pro:** $10/month - Priority access to GPUs
- **Colab Pro+:** $50/month - Guaranteed GPU access
- **Training Time:** ~2-4 hours on T4
- **Total Cost:** Free tier sufficient for most use cases

## 📊 Monitoring Progress

### Check GPU Usage
```python
!nvidia-smi
```

### Check Disk Usage
```python
!df -h
```

### Check Training Progress
```python
!tail -f /content/egyption_id_ready/output/egyptian_id_rec/train.log
```

### View TensorBoard (Optional)
```python
%load_ext tensorboard
%tensorboard --logdir /content/egyption_id_ready/output/egyptian_id_rec/
```

## 🎓 Next Steps After Colab

### 1. Deploy Model
- Download ONNX model from Google Drive
- Integrate with your application
- Use FastAPI endpoint from the project

### 2. Improve Model
- Collect more training data
- Fine-tune hyperparameters
- Try different OCR backbones

### 3. Production Deployment
- Use Docker container from project
- Deploy to cloud (AWS, GCP, Azure)
- Set up monitoring and logging

## 📞 Support

For issues or questions:
1. Check project README.md
2. Review docs/ folder for detailed guides
3. Open GitHub issue

## 📄 License

Same as main project - MIT License

---

**Happy OCR-ing! 🚀**
