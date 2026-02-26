# 🇪🇬 Egyptian ID OCR

نظام OCR متكامل لاستخراج بيانات بطاقة الرقم القومي المصرية — مُحسَّن للعمل على **GPU** للتدريب و **CPU** للإنتاج.

```
ID Card Image → YOLO Detection → Field Cropping → PaddleOCR Recognition → JSON Output
```

---

## ✨ المميزات

- **دعم GPU**: تدريب أسرع بـ 10 أضعاف مع دعم الـ **Checkpoints** والاستئناف (Resume).
- **هيكل موديلار (src)**: كود منظم وسهل التطوير مع فصل محركات الـ OCR (Gemini/QARI).
- **Notebooks تفاعلية**: 3 ملفات `.ipynb` تغطي الـ pipeline بالكامل من قص الصور للتدريب للتقييم.
- **سريع في الإنتاج**: ~17ms لكل حقل باستخدام ONNX Runtime على الـ CPU.
- **24 حقل مدعوم**: الاسم، الرقم القومي، العنوان، المحافظة، الديانة، الحالة الاجتماعية، المهنة، إلخ.
- **API جاهز**: FastAPI مع 5 endpoints + Docker جاهزة للنشر.

---

## 📁 هيكل المشروع

```
egyption_id_ready/
├── src/                    # الكود المصدري (Core Logic)
│   ├── ocr_engines/        # محركات الـ OCR (Gemini & QARI)
│   ├── preprocessing.py    # معالجة الصور وتحسين الجودة
│   ├── label_reader.py     # قراءة YOLO labels (24 class)
│   ├── field_detector.py   # كشف الحقول بـ ONNX
│   ├── crop_builder.py     # قص الحقول من الصور
│   ├── text_cleaner.py     # تنظيف النص العربي وعكس الـ RTL
│   └── inference.py        # Pipeline الاستخدام النهائي (ONNX)
│
├── notebooks/              # Jupyter Notebooks (التسلسل التفاعلي)
│   ├── 01_build_dataset.ipynb   # قص الحقول + فحص الجودة
│   ├── 02_label_and_train.ipynb # استخراج النص + التدريب (GPU)
│   └── 03_evaluate_and_deploy.ipynb # التقييم + ONNX + الـ API
│
├── scripts/                # سكربتات الأتمتة
│   ├── build_dataset.py    # معالجة دفعات الصور وقصها
│   ├── label_crops.py      # استخراج النص التلقائي (Gemini/QARI)
│   ├── prepare_paddle_labels.py # تجهيز داتا التدريب
│   ├── train.sh            # تشغيل التدريب (PaddleOCR)
│   └── export_onnx.sh      # تصدير الموديل لـ ONNX
│
├── configs/                # إعدادات الموديل (YAML)
├── app/                    # واجهة الـ API (FastAPI)
├── tests/                  # الاختبارات التلقائية (15 test)
├── model/                  # ملفات الموديلات (field_detector.onnx)
└── onnx/                   # ملفات الـ OCR النهائية المصدرة
```

---

## 🚀 التشغيل السريع (Pipeline)

أفضل وسيلة لتشغيل المشروع هي عبر الـ **Notebooks** بالترتيب:

### 1. إعداد البيئة
```powershell
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### 2. بناء الداتا (01_build_dataset.ipynb)
- يقوم بقص حقول البطاقة من الصور الأصلية باستخدام الـ Labels.
- يحلل جودة الصور (التباين، الوضوح).
- **Output**: `rec/images/` + `crops_metadata.csv`

### 3. التدريب والاستئناف (02_label_and_train.ipynb)
- **Labeling**: استخراج النص الفعلي باستخدام Gemini API أو QARI-OCR.
- **Training**: يبدأ تدريب PaddleOCR مع دعم الـ **GPU**.
- **Checkpoints**: يتم حفظ الموديل كل 5 epochs وتستطيع الاستكمال من آخر checkpoint بسهولة.

### 4. التصدير والإنتاج (03_evaluate_and_deploy.ipynb)
- يحسب دقة الموديل (CER/WER).
- يصدر الموديل لـ **ONNX** ويقوم بعمل Optimization له.
- يختبر الـ API والـ Benchmark.

---

## 🔌 استخدام الـ API

### تشغيل السيرفر
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### عينة من الـ JSON Response
```json
{
  "status": "success",
  "latency_ms": 245,
  "fields": {
    "name": {"text": "محمد أحمد علي", "confidence": 0.98, "valid": true},
    "national_id": {"text": "29001011234567", "confidence": 0.99, "valid": true}
  }
}
```

---

## 🧪 الاختبارات والجودة

المشروع مغطى بـ **15 اختبار** لوحدات الـ core لضمان استقرار الكود بعد أي تعديل:
```powershell
python -m pytest tests/test_pipeline.py -v
```

---

## ⚙️ التقنيات المستخدمة

- **PaddleOCR (SVTR_LCNet)**: الموديل الأساسي (نحيف وسريع).
- **YOLO v8 (ONNX)**: لكشف حقول البطاقة.
- **ONNX Runtime**: لتشغيل الـ Inference بأقصى سرعة على CPU.
- **FastAPI**: لتقديم الخدمة كـ REST API.
- **Pydantic**: للتحقق من صحة البيانات (Data Validation).

---

## 📝 ملاحظات هامة
- **التدريب**: يفضل وجود NVIDIA GPU (VRAM 6GB+).
- **الاسترجاع**: لو توقف التدريب، ملف الـ `configs/egyptian_id_rec.yml` مهيأ لاستكمال التدريب من آخر Checkpoint.
- **دعم اللغة**: يدعم النصوص العربية والأرقام بدقة عالية جداً.

---
📄 **رخصة**: MIT License
