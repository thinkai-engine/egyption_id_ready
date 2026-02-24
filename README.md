# 🇪🇬 Egyptian ID OCR

نظام OCR متكامل لاستخراج بيانات بطاقة الرقم القومي المصرية — مُحسَّن للعمل على **CPU** بدون GPU.

```
ID Card Image → YOLO Detection → Field Cropping → PaddleOCR Recognition → JSON Output
```

---

## ✨ المميزات

- **سريع على CPU**: ~17ms per field باستخدام ONNX Runtime
- **دقة عالية**: Fine-tuned PaddleOCR (SVTR_LCNet) على بيانات مصرية حقيقية
- **24 حقل مدعوم**: الاسم، الرقم القومي، تاريخ الميلاد، العنوان، المحافظة، الديانة، الحالة الاجتماعية، المهنة...
- **API جاهز**: FastAPI مع 5 endpoints + Docker
- **تحقق ذكي**: Validation rules لكل حقل (مثلاً: الرقم القومي = 14 رقم يبدأ بـ 2 أو 3)

---

## 📁 هيكل المشروع

```
egyptian_id_ready/
├── preprocessing.py          # معالجة الصور (deskew, contrast, denoise)
├── label_reader.py           # قراءة YOLO labels (24 class)
├── field_detector.py         # كشف الحقول بـ ONNX (YOLOFieldDetector)
├── crop_builder.py           # قص الحقول من الصور
├── gemini_ocr.py             # استخراج النص بـ Gemini API
├── qari_ocr.py               # استخراج النص بـ QARI-OCR (HuggingFace)
├── text_cleaner.py           # تنظيف النص العربي + عكس RTL
├── inference.py              # Pipeline الإنتاج (ONNX)
├── arabic_dict.txt           # قاموس الحروف (74 حرف)
├── requirements.txt          # المكتبات المطلوبة
├── Dockerfile                # Docker image
├── docker-compose.yml        # Docker deployment
│
├── app/
│   └── main.py               # FastAPI (5 endpoints)
│
├── configs/
│   └── egyptian_id_rec.yml   # إعدادات PaddleOCR للتدريب
│
├── scripts/
│   ├── build_dataset.py      # قص كل الحقول من كل الصور
│   ├── label_crops.py        # استخراج النص (qari|gemini|both)
│   ├── prepare_paddle_labels.py  # إنشاء ملفات train/val/test
│   ├── evaluate.py           # تقييم (CER/WER/Exact Match)
│   ├── benchmark.py          # قياس السرعة على CPU
│   ├── train.sh              # تشغيل التدريب
│   ├── export_onnx.sh        # تصدير ONNX
│   └── run_full_pipeline.sh  # تشغيل كل شيء
│
├── tests/
│   └── test_pipeline.py      # 12 اختبار
│
├── model/
│   └── field_detector.onnx   # نموذج YOLO لكشف الحقول
│
├── train/                    # 15,669 صورة + labels
├── valid/                    # 948 صورة + labels
└── test/                     # 103 صورة + labels
```

---

## 🚀 التشغيل السريع

### 1. إعداد البيئة

```powershell
cd k:\business\egyption_id_ready
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. بناء الـ Dataset

```powershell
python scripts/build_dataset.py
```
**Output**: `rec/images/` (حقول مقصوصة) + `crops_metadata.csv`

### 3. استخراج النصوص (Labeling)

```powershell
# Option A: Gemini API (يحتاج إنترنت + API key)
python scripts/label_crops.py --method gemini --gemini-key YOUR_KEY

# Option B: QARI-OCR (يحتاج GPU)
python scripts/label_crops.py --method qari

# Option C: المقارنة بين الاثنين
python scripts/label_crops.py --method both --gemini-key YOUR_KEY
```
**Output**: `crops_labeled.csv`

### 4. تجهيز ملفات التدريب

```powershell
python scripts/prepare_paddle_labels.py
```
**Output**: `rec/train.txt`, `rec/val.txt`, `rec/test.txt`

### 5. Fine-tuning

```powershell
pip install paddlepaddle paddleocr
git clone https://github.com/PaddlePaddle/PaddleOCR
cd PaddleOCR
python tools/train.py -c ../configs/egyptian_id_rec.yml
```
⏱️ **المدة**: ~8-12 ساعة على CPU

### 6. تصدير ONNX

```powershell
pip install paddle2onnx onnxsim
python tools/export_model.py -c ../configs/egyptian_id_rec.yml ^
    -o Global.pretrained_model=../output/egyptian_id_rec/best_accuracy ^
       Global.save_inference_dir=../inference/rec
paddle2onnx --model_dir ../inference/rec --save_file ../onnx/rec.onnx --opset_version 11
python -m onnxsim ../onnx/rec.onnx ../onnx/rec_sim.onnx
```

### 7. تقييم

```powershell
cd ..
python scripts/evaluate.py
```

### 8. تشغيل الـ API

```powershell
# مباشر
uvicorn app.main:app --host 0.0.0.0 --port 8000

# أو Docker
docker compose up --build -d
```
🌐 **Swagger UI**: http://localhost:8000/docs

---

## 🔌 API Endpoints

| Method | Endpoint | الوصف |
|--------|----------|-------|
| `GET`  | `/health` | حالة السيرفر |
| `POST` | `/ocr/full-card` | استخراج كل الحقول من صورة بطاقة كاملة |
| `POST` | `/ocr/single-field` | استخراج نص حقل واحد (صورة مقصوصة) |
| `POST` | `/ocr/batch` | معالجة دفعة (حتى 20 صورة) |
| `GET`  | `/metrics` | إحصائيات الاستخدام |

### مثال استخدام

```python
import requests

# استخراج كل حقول البطاقة
with open("id_card.jpg", "rb") as f:
    r = requests.post(
        "http://localhost:8000/ocr/full-card",
        files={"file": ("id.jpg", f, "image/jpeg")}
    )

print(r.json())
# {
#   "status": "success",
#   "latency_ms": 245.3,
#   "fields": {
#     "name":        {"text": "محمد أحمد علي", "confidence": 0.967, "valid": true},
#     "national_id": {"text": "29001011234567", "confidence": 0.994, "valid": true},
#     ...
#   }
# }
```

```bash
# أو بـ curl
curl -X POST http://localhost:8000/ocr/single-field \
  -F "file=@name_crop.jpg" \
  -F "field=name"
```

---

## 🧪 الاختبارات

```powershell
python -m pytest tests/test_pipeline.py -v
```

| الاختبار | المكون | ما يغطيه |
|---|---|---|
| `test_parse_yolo_label_*` | label_reader | تحويل الإحداثيات |
| `test_crop_field_*` | crop_builder | قص + padding + edge cases |
| `test_clean_arabic_*` | text_cleaner | تنظيف + RTL reversal |
| `test_field_preprocessor_*` | preprocessing | output shape |
| `test_validate_national_id_*` | inference | validation rules |
| `test_health_endpoint` | API | health check |

---

## 📊 حقول البطاقة المدعومة (24 حقل)

| Class ID | الحقل | Class ID | الحقل |
|----------|-------|----------|-------|
| 0 | المهنة | 7 | الرقم القومي |
| 1 | الصورة | 8 | الحالة الزوجية |
| 2 | تاريخ الانتهاء | 9 | الجنس |
| 3 | تاريخ الميلاد | 10 | المحافظة |
| 4 | الديانة | 11 | اسم الزوج |
| 5 | الاسم | 12 | تاريخ الإصدار |
| 6 | العنوان | 23 | الرقم التسلسلي |

---

## ⚙️ التقنيات المستخدمة

| المكون | التقنية | لماذا |
|--------|---------|-------|
| Detection | YOLO (ONNX) | سريع ودقيق لكشف الحقول |
| Recognition | PaddleOCR PP-OCRv3 Arabic | ~10MB، مُحسَّن للـ CPU |
| Architecture | MobileNetV1Enhance + SVTR | أفضل أداء/حجم للـ mobile |
| Inference | ONNX Runtime | ~17ms per field على CPU |
| API | FastAPI | سريع وasync ready |
| Deployment | Docker | Portable + healthcheck |

---

## 📝 ملاحظات

- **بدون GPU**: كل الـ inference يعمل على CPU بكفاءة. التدريب بطيء على CPU (~12 ساعة) لكنه يعمل.
- **الدقة المتوقعة**: CER < 5% للحقول الواضحة (الاسم، الرقم القومي). العنوان أصعب بسبب طوله.
- **الحقل 1 (الصورة)**: يتم تجاهله تلقائياً لأنه ليس نصاً.

---

## 📄 الرخصة

MIT License
