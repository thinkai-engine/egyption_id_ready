# 🇪🇬 Egyptian ID OCR

A complete OCR system for extracting data from Egyptian National ID cards — optimized for **GPU training** and **CPU production deployment**.

```
ID Card Image → YOLO Detection → Field Cropping → PaddleOCR Recognition → JSON Output
```

---

## ✨ Features

- **GPU Acceleration**: 10x faster training with **checkpoint** support and resume capability
- **Modular Architecture**: Clean, maintainable code with separated OCR engines (Gemini/QARI)
- **Interactive Notebooks**: 3 Jupyter notebooks covering the complete pipeline from image cropping to training and evaluation
- **Production Ready**: ~17ms per field using ONNX Runtime on CPU
- **24 Supported Fields**: Name, National ID, Address, Governorate, Religion, Marital Status, Profession, and more
- **Ready-to-Use API**: FastAPI with 5 endpoints + Docker deployment support

---

## 📁 Project Structure

```
egyption_id_ready/
├── src/                    # Core source code
│   ├── ocr_engines/        # OCR engines (Gemini & QARI)
│   ├── preprocessing.py    # Image processing and enhancement
│   ├── label_reader.py     # YOLO labels reader (24 classes)
│   ├── field_detector.py   # Field detection with ONNX
│   ├── crop_builder.py     # Field cropping from images
│   ├── text_cleaner.py     # Arabic text cleaning and RTL handling
│   └── inference.py        # Final inference pipeline (ONNX)
│
├── notebooks/              # Interactive Jupyter Notebooks
│   ├── 01_build_dataset.ipynb   # Field cropping + quality analysis
│   ├── 02_label_and_train.ipynb # Text extraction + model training
│   └── 03_evaluate_and_deploy.ipynb # Evaluation + ONNX export + API
│
├── scripts/                # Automation scripts
│   ├── build_dataset.py    # Batch image processing and cropping
│   ├── label_crops.py      # Automatic text extraction (Gemini/QARI)
│   ├── prepare_paddle_labels.py # Training data preparation
│   ├── train.sh            # Training execution (PaddleOCR)
│   └── export_onnx.sh      # Model export to ONNX
│
├── configs/                # Model configurations (YAML)
├── app/                    # API server (FastAPI)
├── tests/                  # Automated tests (15 tests)
├── model/                  # Model files (field_detector.onnx)
└── onnx/                   # Exported ONNX OCR models
```

---

## 🚀 Quick Start (Pipeline)

The recommended way to run this project is through the **Notebooks** in sequence:

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\Activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Build Dataset (01_build_dataset.ipynb)

This notebook:
- Crops card fields from original images using YOLO labels
- Analyzes image quality (contrast, sharpness, brightness)
- Generates metadata CSV for tracking

**Outputs**: `rec/images/` + `crops_metadata.csv`

### 3. Labeling and Training (02_label_and_train.ipynb)

This notebook handles:
- **Labeling**: Extract ground truth text using Gemini API or QARI-OCR
- **Training**: Fine-tune PaddleOCR with GPU acceleration
- **Checkpoints**: Model saved every 5 epochs with easy resume capability

### 4. Export and Deployment (03_evaluate_and_deploy.ipynb)

This notebook:
- Calculates model accuracy (CER/WER metrics)
- Exports model to **ONNX** format with optimization
- Tests the API and runs benchmarks

---

## 🔌 API Usage

### Start the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/ocr` | POST | Extract all fields from ID card image |
| `/ocr/field/{field_name}` | POST | Extract specific field only |
| `/validate/national_id` | POST | Validate Egyptian National ID format |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

### Example Request

```bash
curl -X POST "http://localhost:8000/ocr" \
  -F "file=@id_card.jpg"
```

### Example JSON Response

```json
{
  "status": "success",
  "latency_ms": 245,
  "fields": {
    "name": {
      "text": "محمد أحمد علي",
      "confidence": 0.98,
      "valid": true
    },
    "national_id": {
      "text": "29001011234567",
      "confidence": 0.99,
      "valid": true
    },
    "address": {
      "text": "القاهرة - مصر الجديدة",
      "confidence": 0.95,
      "valid": true
    },
    "governorate": {
      "text": "القاهرة",
      "confidence": 0.97,
      "valid": true
    }
  }
}
```

### Python Client Example

```python
import requests

response = requests.post(
    "http://localhost:8000/ocr",
    files={"file": open("id_card.jpg", "rb")}
)
data = response.json()

for field_name, field_data in data["fields"].items():
    print(f"{field_name}: {field_data['text']} ({field_data['confidence']:.2f})")
```

---

## 🧪 Testing and Quality Assurance

The project includes **15 automated tests** covering core units to ensure stability after any modifications:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_pipeline.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

---

## ⚙️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **OCR Engine** | PaddleOCR (SVTR_LCNet) | Primary recognition model (lightweight & fast) |
| **Detection** | YOLO v8 (ONNX) | Card field detection and localization |
| **Inference** | ONNX Runtime | High-speed CPU inference |
| **API Server** | FastAPI | RESTful API service |
| **Validation** | Pydantic | Data validation and serialization |
| **Alternative OCR** | QARI-OCR (Qwen2-VL-2B) | High-accuracy Arabic OCR |
| **Alternative OCR** | Gemini Vision | Google's multimodal API |

---

## 📊 Supported Fields

The system extracts 24 different fields from Egyptian National ID cards:

| Field Name (Arabic) | Field Name (English) | Type |
|---------------------|---------------------|------|
| الاسم | Name | Text |
| الرقم القومي | National ID | Numeric (14 digits) |
| العنوان | Address | Text |
| المحافظة | Governorate | Categorical |
| الدين | Religion | Categorical |
| الحالة الاجتماعية | Marital Status | Categorical |
| المهنة | Profession | Text |
| تاريخ الميلاد | Date of Birth | Date |
| النوع | Gender | Categorical |
| تاريخ الانقضاء | Expiry Date | Date |

---

## 🐳 Docker Deployment

### Build the Image

```bash
docker build -t egyptian-id-ocr .
```

### Run the Container

```bash
docker run -d -p 8000:8000 --name ocr-api egyptian-id-ocr
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 📈 Performance Benchmarks

| Metric | Value | Hardware |
|--------|-------|----------|
| **Inference Time (per field)** | ~17ms | CPU (Intel i7) |
| **Inference Time (full card)** | ~245ms | CPU (Intel i7) |
| **Training Speed** | ~1000 samples/sec | GPU (RTX 3060) |
| **Character Error Rate (CER)** | < 5% | Test set |
| **Word Error Rate (WER)** | < 8% | Test set |

---

## 📝 Important Notes

### Training Requirements

- **GPU**: NVIDIA GPU with 6GB+ VRAM recommended (RTX 3060 or better)
- **RAM**: 16GB system RAM minimum
- **Storage**: 10GB free space for datasets and checkpoints

### Resuming Training

If training is interrupted, the configuration file `configs/egyptian_id_rec.yml` is pre-configured to resume from the last checkpoint:

```yaml
Global:
  checkpoints: ./output/egyptian_id_rec/latest
```

### Language Support

- **Primary**: Arabic text with high accuracy
- **Secondary**: Arabic-Indic numerals (٠١٢٣٤٥٦٧٨٩)
- **Tertiary**: Western Arabic numerals (0123456789)

### Model Quantization

For production deployment with limited resources:

```python
# 4-bit quantization (requires bitsandbytes)
qari = QariOCR(use_4bit=True)  # Use if VRAM < 6GB

# Standard loading
qari = QariOCR(use_4bit=False)  # Recommended for 6GB+ VRAM
```

---

## 🔧 Troubleshooting

### Common Issues

**1. CUDA/GPU Errors**
```
UserWarning: CUDA initialization error
```
- Ensure NVIDIA drivers are installed correctly
- Check CUDA toolkit version compatibility
- Try running with `use_gpu=False` for CPU-only mode

**2. Model Loading Errors**
```
ImportError: bitsandbytes required for 4-bit quantization
```
- Install bitsandbytes: `pip install -U bitsandbytes>=0.46.1`
- Or use `use_4bit=False` in model initialization

**3. Low Accuracy**
- Verify dataset quality and label correctness
- Ensure Arabic text is properly reversed (RTL → LTR)
- Check that all characters exist in `arabic_dict.txt`

**4. Memory Issues**
- Reduce `batch_size` in config file
- Use gradient accumulation
- Enable mixed precision training

---

## 📚 Additional Documentation

- [Implementation Plan](implementation_plan.md) - Detailed technical specifications
- [Configuration Guide](configs/README.md) - YAML configuration reference
- [API Documentation](app/README.md) - Complete API reference

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **PaddlePaddle** for the excellent OCR framework
- **NAMAA-Space** for the QARI-OCR Arabic recognition model
- **Ultralytics** for YOLO detection capabilities
- **Google** for Gemini Vision API
- The open-source community for various Arabic NLP tools

---

## 📬 Contact

For questions, issues, or feature requests, please open an issue on GitHub.

---

**Built with ❤️ for Arabic OCR**
