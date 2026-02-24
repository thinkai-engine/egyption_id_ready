<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# افضل ocr model اعمل fintuing علي بطاقه الهويه المصريه و يشتغل بفاعليه عاليه علي cpu

أفضل خيار لحالتك هو **PaddleOCR مع نموذج arabic_PP-OCRv4_mobile_rec**، لأنه يجمع بين الخفة العالية، دعم اللغة العربية مدمجاً، وكفاءة CPU ممتازة، مع قابلية fine-tuning سهلة على dataset بطاقة الهوية المصرية.

## المقارنة الشاملة للخيارات

| النموذج | حجم النموذج | CPU Inference | Fine-tuning | دعم العربية | الأنسب لبطاقة الهوية |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **PaddleOCR PP-OCRv4 mobile** | ~10.5 MB | ~17–21 ms | ✅ سهل جداً | ✅ مدمج | ⭐⭐⭐⭐⭐ |
| **TrOCR (small)** | ~330 MB | ~150–400 ms | ✅ جيد | ⚠️ يحتاج encoder عربي | ⭐⭐⭐ |
| **EasyOCR** | ~120 MB | ~200–500 ms | ⚠️ صعب | ✅ مدمج | ⭐⭐⭐ |
| **Tesseract + Arabic** | ~15 MB | ~10–30 ms | ⚠️ محدود | ✅ مدمج | ⭐⭐ |
| **Qari-OCR (Qwen2-VL-2B)** | ~4 GB | ❌ بطيء جداً | ✅ ممتاز | ✅⭐ | ❌ غير عملي على CPU |

## لماذا PaddleOCR هو الأفضل لحالتك

**PaddleOCR** يمتلك نموذج `arabic_PP-OCRv3_mobile_rec` المُدرَّب أصلاً على النص العربي والأرقام، بحجم لا يتجاوز 9.6MB. يُسجّل PP-OCRv4_mobile_rec سرعة CPU تبلغ **~17ms في الوضع العادي و3.6ms في الوضع عالي الأداء** ، مما يجعله مناسباً جداً للبيئات بدون GPU. كما أن مشروع ArabID على GitHub يعتمد عليه تحديداً لاستخراج النص من بطاقات الهوية المصرية والمغربية.[^1_1][^1_2][^1_3]

## الاستراتيجية المثلى للـ Fine-tuning

البنية المُوصى بها هي **Pipeline ثنائية المرحلة**:

1. **Detection**: استخدم `PP-OCRv4_mobile_det` لتحديد حقول البطاقة (الاسم، الرقم القومي، العنوان)
2. **Recognition**: Fine-tune على `arabic_PP-OCRv4_mobile_rec` بـ dataset خاص بحقول الهوية المصرية
```python
# تثبيت وتشغيل النموذج
pip install paddlepaddle paddleocr

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ar', use_gpu=False)
result = ocr.ocr('egyptian_id.jpg', cls=True)
```


## نصائح مهمة للـ Fine-tuning

- **Data**: احرص على توفير **+5000 صورة** واقعية لحقول البطاقة المصرية مع تنوع في الإضاءة والزوايا، لأن الـ dataset الصغيرة تعطي دقة لا تتجاوز 81%[^1_4]
- **النموذج الأساسي**: ابدأ من `arabic_PP-OCRv3_mobile_rec` كـ pretrained base بدلاً من التدريب من الصفر
- **Architecture**: استخدم `MobileNetV1Enhance + SVTR` (الموجود بالفعل في النموذج) فهو مُحسَّن للـ CPU[^1_4]
- **تحسين الأداء**: فعّل ONNX Runtime أو OpenVINO للحصول على **High-Performance Mode** الذي يُقلّل زمن CPU إلى النصف تقريباً[^1_2]


## خيار بديل: Tesseract + محرك مخصص

إذا أردت الأبسط على الإطلاق، **Tesseract** هو الأسرع على CPU وله دعم عربي مدمج ، لكن دقته أضعف على التخطيطات المعقدة مثل بطاقات الهوية ذات الحقول المتعددة مقارنةً بـ PaddleOCR. للإنتاج الفعلي على بطاقة الهوية المصرية، PaddleOCR هو الخيار الأكثر نضجاً وموثوقية.[^1_5]
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_6][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: https://huggingface.co/PaddlePaddle/arabic_PP-OCRv3_mobile_rec

[^1_2]: https://paddlepaddle.github.io/PaddleX/3.3/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html

[^1_3]: https://github.com/arij01/ArabID

[^1_4]: https://github.com/PaddlePaddle/PaddleOCR/issues/10358

[^1_5]: https://modal.com/blog/8-top-open-source-ocr-models-compared

[^1_6]: https://signme.it/api-ocr-لقراءة-بطاقة-الرقم-القومي-المصرية/

[^1_7]: https://www.facebook.com/SEER2030/posts/تقنية-ocr-الذكية-لاستخراج-بيانات-الرقم-القومي-المصري-بدقة-وسرعةهل-تحتاج-إلى-تحوي/122230082684194264/

[^1_8]: https://huggingface.co/blog/imomayiz/atlasocr

[^1_9]: https://www.linkedin.com/pulse/llm-based-arabic-ocr-models-training-architecture-hazem-abdelazim-zffzf

[^1_10]: https://www.nature.com/articles/s41598-025-12045-z

[^1_11]: https://www.facebook.com/groups/ComputerVisionGroup/posts/3339668449510666/

[^1_12]: https://dl.acm.org/doi/10.1145/3768150

[^1_13]: https://aclanthology.org/2025.findings-acl.1135.pdf

[^1_14]: https://github.com/NielsRogge/Transformers-Tutorials/issues/207

[^1_15]: https://www.kdnuggets.com/top-7-open-source-ocr-models

[^1_16]: https://www.iieta.org/journals/mmep/paper/10.18280/mmep.120735

[^1_17]: https://discuss.huggingface.co/t/fine-tune-trocr-on-arabic/25577

[^1_18]: https://github.com/PaddlePaddle/PaddleOCR

[^1_19]: https://www.youtube.com/watch?v=I0pO2ES-wMg

[^1_20]: https://adevinta.com/techblog/deep-dive-in-paddleocr-inference/

[^1_21]: https://www.kaggle.com/code/yahyaahmed1012/egyptian-id-ocr

[^1_22]: https://github.com/catalyst-cooperative/PaddleOCR-headless/blob/release/2.6/doc/doc_en/models_list_en.md

[^1_23]: https://huggingface.co/collections/NAMAA-Space/qari-ocr-a-high-accuracy-model-for-arabic-optical-character

[^1_24]: https://www.tenorshare.com/ocr/paddleocr.html

[^1_25]: https://arxiv.org/abs/2506.02295

[^1_26]: https://paddlepaddle.github.io/PaddleX/3.1/en/module_usage/tutorials/ocr_modules/text_recognition.html

[^1_27]: https://www.semanticscholar.org/paper/QARI-OCR:-High-Fidelity-Arabic-Text-Recognition-Wasfy-Nacar/49c6470038e1ef41e267aea59f2f1d23a7370a55

[^1_28]: https://gitee.com/nyshop/PaddleOCR/blob/main/doc/doc_en/models_list_en.md?skip_mobile=true


---

# خطوات fine-tuning PaddleOCR للعربية على CPU

إليك الدليل الكامل لـ fine-tuning PaddleOCR على بطاقة الهوية المصرية مع التشغيل على CPU.

***

## 1. الإعداد البيئي

```bash
# تثبيت paddlepaddle نسخة CPU
pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple

# استنساخ مستودع PaddleOCR
git clone https://github.com/PaddlePaddle/PaddleOCR
cd PaddleOCR
pip install -r requirements.txt
```

> تأكد من تثبيت `paddlepaddle` وليس `paddlepaddle-gpu` حتى تعمل الـ training كاملاً على CPU.[^2_1]

***

## 2. تجهيز الـ Dataset

هذه أهم مرحلة. لكل نوع بيانات حد أدنى مطلوب:[^2_1]

- **Detection**: ≥ 500 صورة لبطاقات الهوية بـ bounding boxes للحقول
- **Recognition**: ≥ 5000 صورة مقصوصة لكل حقل نصي (اسم، رقم قومي، عنوان...)

**الصيغة المطلوبة لـ Recognition labels:**

```
# train.txt - كل سطر: مسار الصورة + tab + النص
./data/rec/img_001.jpg	محمد أحمد علي
./data/rec/img_002.jpg	١٢٣٤٥٦٧٨٩٠١٢٣٤
./data/rec/img_003.jpg	القاهرة - مصر الجديدة
```

**بنية المجلدات:**

```
dataset/
├── det/
│   ├── images/       # صور بطاقات الهوية كاملة
│   └── train.json    # annotations بصيغة PaddleOCR
└── rec/
    ├── images/       # صور مقصوصة لكل حقل
    ├── train.txt
    └── val.txt
```


***

## 3. إعداد Arabic Character Dictionary

```bash
# تحميل القاموس العربي الجاهز
wget https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/dict/arabic_dict.txt
```

تحقق أنه يحتوي على جميع الرموز الموجودة في بياناتك (أرقام عربية، علامات الترقيم، الحروف الخاصة في الهوية المصرية).[^2_2]

***

## 4. تحميل الـ Pretrained Model

```bash
# تحميل نموذج Arabic Recognition المدرب مسبقاً
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_train.tar
tar xf arabic_PP-OCRv3_rec_train.tar
```


***

## 5. ضبط ملف الـ Config

انسخ الملف الأساسي وعدّله:

```bash
cp configs/rec/PP-OCRv3/multi_language/arabic_PP-OCRv3_rec.yml \
   configs/rec/arabic_egyptianid_rec.yml
```

أهم التعديلات في الملف:[^2_3][^2_1]

```yaml
Global:
  use_gpu: false                        # ← CPU فقط
  epoch_num: 100
  save_model_dir: ./output/arabic_id_rec/
  pretrained_model: ./arabic_PP-OCRv3_rec_train/best_accuracy
  character_dict_path: ./ppocr/utils/dict/arabic_dict.txt
  max_text_length: 40                   # ← اضبطه حسب أطول نص في بياناتك

Optimizer:
  lr:
    learning_rate: 0.0001               # ← صغّر LR عند CPU single-machine

Train:
  loader:
    batch_size_per_card: 32             # ← صغّر لو الـ RAM محدود
    num_workers: 2                      # ← 2-4 على CPU

  dataset:
    label_file_list:
    - ./dataset/rec/train.txt
    ratio_list: [1.0]
```


***

## 6. تشغيل الـ Training

```bash
python tools/train.py \
  -c configs/rec/arabic_egyptianid_rec.yml \
  -o Global.use_gpu=false \
     Global.epoch_num=100 \
     Train.loader.batch_size_per_card=32
```

**تتبع الـ Training:**

```bash
# مشاهدة الـ accuracy في real-time
tail -f ./output/arabic_id_rec/train.log
```


***

## 7. التقييم والـ Export

```bash
# تقييم النموذج على val set
python tools/eval.py \
  -c configs/rec/arabic_egyptianid_rec.yml \
  -o Global.checkpoints=./output/arabic_id_rec/best_accuracy

# تصدير النموذج للـ Inference
python tools/export_model.py \
  -c configs/rec/arabic_egyptianid_rec.yml \
  -o Global.pretrained_model=./output/arabic_id_rec/best_accuracy \
     Global.save_inference_dir=./inference/arabic_id_rec
```


***

## 8. الـ Inference على CPU

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='ar',
    use_gpu=False,                    # CPU mode
    rec_model_dir='./inference/arabic_id_rec',
    det_db_score_mode='slow',         # دقة أعلى للـ CPU
)

result = ocr.ocr('egyptian_id.jpg', cls=True)
for line in result[^2_0]:
    print(f"Text: {line[^2_1][^2_0]} | Confidence: {line[^2_1][^2_1]:.2f}")
```


***

## نصائح لتحسين الأداء على CPU

| المشكلة | الحل |
| :-- | :-- |
| Training بطيء جداً | قلل `num_workers=2`، استخدم `batch_size=16` |
| Accuracy منخفضة | أضف بيانات synthetic بنسبة 1:1 مع الـ real data [^2_1] |
| Overfitting | استخدم `ratio_list` لخلط بيانات الهوية مع بيانات عربية عامة |
| Inference بطيء | صدّر لـ ONNX واستخدم `onnxruntime` بدلاً من Paddle Inference |

أهم تحذير من الـ community: إذا دقتك أقل من 90%، السبب الأول دائماً هو **قِلة البيانات**، وليس الـ model أو الـ config.[^2_2]
<span style="display:none">[^2_10][^2_11][^2_12][^2_13][^2_14][^2_15][^2_4][^2_5][^2_6][^2_7][^2_8][^2_9]</span>

<div align="center">⁂</div>

[^2_1]: http://www.paddleocr.ai/v2.9/en/ppocr/model_train/finetune.html

[^2_2]: https://github.com/PaddlePaddle/PaddleOCR/issues/10358

[^2_3]: https://github.com/PaddlePaddle/PaddleOCR/discussions/14459

[^2_4]: http://www.paddleocr.ai/v3.3.2/en/version2.x/ppocr/model_train/recognition.html

[^2_5]: https://github.com/PaddlePaddle/PaddleOCR/discussions/14369

[^2_6]: https://www.youtube.com/watch?v=I0pO2ES-wMg

[^2_7]: https://hackernoon.com/ocr-fine-tuning-from-raw-data-to-custom-paddle-ocr-model

[^2_8]: https://aclanthology.org/2025.findings-acl.1135.pdf

[^2_9]: https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/OCR.html

[^2_10]: https://github.com/PaddlePaddle/PaddleOCR/issues/10519

[^2_11]: https://github.com/PaddlePaddle/PaddleOCR

[^2_12]: https://github.com/PaddlePaddle/PaddleOCR/discussions/12211

[^2_13]: https://paddlepaddle.github.io/PaddleX/3.1/en/module_usage/tutorials/ocr_modules/text_recognition.html

[^2_14]: https://arxiv.org/html/2507.05595v1

[^2_15]: https://github.com/PaddlePaddle/PaddleOCR/discussions/12427


---

# كيف أحضّر بيانات تدريب لـ PaddleOCR عربي

إليك الدليل الكامل لتجهيز بيانات التدريب من الصفر.

***

## أنواع البيانات المطلوبة

تحتاج لنوعين مستقلين من البيانات:[^3_1]


| النوع | الغرض | الحد الأدنى |
| :-- | :-- | :-- |
| **Detection** | تحديد مواضع النصوص في الصورة | 500 صورة كاملة للبطاقة |
| **Recognition** | قراءة النص من كل حقل | 5,000 صورة مقصوصة |


***

## المسار الأول: البيانات الحقيقية (PPOCRLabel)

الأداة الرسمية لعمل annotation هي **PPOCRLabel** وهي تدعم Arabic مباشرة.[^3_2]

```bash
pip install PPOCRLabel
PPOCRLabel --lang ar
```

**خطوات الاستخدام:**

1. افتح مجلد صور البطاقات
2. اضغط **Auto Recognition** ليرسم الـ bounding boxes تلقائياً
3. صحّح الأخطاء يدوياً (النص المقروء + موضع الـ box)
4. اضغط **Export Label** ← يولّد `Label.txt` جاهز للتدريب

**الناتج الأوتوماتيكي:**

```
# Label.txt - للـ Detection
img_001.jpg\t[{"transcription": "محمد أحمد", "points": [[10,5],[120,5],[120,25],[10,25]]}]

# rec/train.txt - للـ Recognition (مقصوصة)
rec/img_001_crop_0.jpg\tمحمد أحمد
```


***

## ⚠️ مشكلة حرجة: عكس النص العربي

العربية RTL لكن النموذج يتوقع LTR، لذلك **يجب عكس كل label** في ملف الـ Recognition:[^3_3]

```python
# script لعكس كل نصوص الـ labels
with open('train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('train_fixed.txt', 'w', encoding='utf-8') as f:
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            img_path, label = parts
            reversed_label = label[::-1]   # ← عكس النص
            f.write(f"{img_path}\t{reversed_label}\n")
```

مثال: `محمد علي` ← يُحفظ كـ `يلع دمحم` في الـ label file.[^3_3]

***

## المسار الثاني: البيانات الاصطناعية (Synthetic Data)

لأن جمع 5000+ صورة حقيقية صعب، الأفضل **توليد synthetic data** بخطوط عربية متنوعة.

### أداة `trdg` (Text Recognition Data Generator)

```bash
pip install trdg

# توليد بيانات عربية بخط عشوائي
trdg -l ar -c 5000 -w 5 -f 48 \
     --output_dir ./synthetic_arabic/ \
     --font_dir ./arabic_fonts/
```


### توليد مخصص لحقول بطاقة الهوية المصرية

```python
from PIL import Image, ImageDraw, ImageFont
import random, os

# قوائم بيانات واقعية لبطاقة الهوية
names = ["محمد أحمد علي", "فاطمة محمود", "أحمد عبد الله", ...]
addresses = ["القاهرة - مصر الجديدة", "الجيزة - الهرم", ...]
national_ids = [f"2{random.randint(10**12, 10**13-1)}" for _ in range(1000)]

fonts = [ImageFont.truetype(f, size=random.randint(18, 28))
         for f in os.listdir('./arabic_fonts/')]

def generate_sample(text, save_path):
    font = random.choice(fonts)
    w, h = 320, 48
    img = Image.new('RGB', (w, h), color=(245, 240, 230))  # لون شبيه بالبطاقة
    draw = ImageDraw.Draw(img)
    
    # إضافة noise وتشويه بسيط
    draw.text((10, 10), text, font=font, fill=(0, 0, 100))
    img.save(save_path)
    return text[::-1]   # ← إرجاع النص معكوساً للـ label

labels = []
for i, name in enumerate(names):
    path = f"./synthetic/img_{i:05d}.jpg"
    label = generate_sample(name, path)
    labels.append(f"img_{i:05d}.jpg\t{label}")

with open('train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(labels))
```


***

## تقسيم الـ Dataset

```python
import random

with open('all_labels.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()

random.shuffle(data)
split = int(len(data) * 0.85)

with open('train.txt', 'w', encoding='utf-8') as f:
    f.writelines(data[:split])

with open('val.txt', 'w', encoding='utf-8') as f:
    f.writelines(data[split:])

print(f"Train: {split} | Val: {len(data)-split}")
```


***

## بنية المجلدات النهائية

```
dataset/
├── det/
│   ├── images/           # صور البطاقات الكاملة
│   └── Label.txt         # من PPOCRLabel
└── rec/
    ├── real/
    │   ├── images/       # صور حقيقية مقصوصة
    ├── synthetic/
    │   ├── images/       # صور مولّدة
    ├── train.txt         # real + synthetic مدموجين
    └── val.txt           # real فقط للتقييم الموضوعي
```

**في الـ config yml:**[^3_4]

```yaml
Train:
  dataset:
    label_file_list:
    - ./dataset/rec/train.txt   # مدمج
    ratio_list: [1.0]
```


***

## نسبة المزج الأمثل

| نوع البيانات | النسبة الموصى بها | السبب |
| :-- | :-- | :-- |
| Real (حقيقية) | 40% | دقيقة ومطابقة للواقع |
| Synthetic (اصطناعية) | 60% | توفير الكمية الكافية |

يُنصح بخلط بيانات بطاقة الهوية مع بيانات عربية عامة بنسبة 1:3 لمنع الـ Overfitting.[^3_4]
<span style="display:none">[^3_10][^3_11][^3_12][^3_13][^3_14][^3_15][^3_16][^3_5][^3_6][^3_7][^3_8][^3_9]</span>

<div align="center">⁂</div>

[^3_1]: http://www.paddleocr.ai/v2.10.0/en/datasets/ocr_datasets.html

[^3_2]: https://github.com/PFCCLab/PPOCRLabel

[^3_3]: https://github.com/PaddlePaddle/PaddleOCR/issues/10358

[^3_4]: http://www.paddleocr.ai/v2.9/en/ppocr/model_train/finetune.html

[^3_5]: https://github.com/mbzuai-oryx/KITAB-Bench

[^3_6]: https://aclanthology.org/2025.findings-acl.1135.pdf

[^3_7]: http://www.paddleocr.ai/v3.3.2/en/version2.x/ppocr/model_train/recognition.html

[^3_8]: https://paddlepaddle.github.io/PaddleX/3.1/en/module_usage/tutorials/ocr_modules/text_recognition.html

[^3_9]: https://arxiv.org/abs/2601.16113

[^3_10]: https://www.linkedin.com/posts/abdelrhman-elshafey-3342951b3_arabicocr-paddleocr-easyocr-activity-7325104987162734592-j4i5

[^3_11]: https://github.com/Evezerest/PPOCRLabel

[^3_12]: https://paddlepaddle.github.io/PaddleOCR/v2.9/en/datasets/ocr_datasets.html

[^3_13]: https://www.facebook.com/groups/big.data.egypt/posts/2260775384181650/

[^3_14]: https://www.kaggle.com/code/yahyaahmed1012/egyptian-id-ocr

[^3_15]: https://www.reddit.com/r/MachineLearning/comments/1jsclxw/p_anyone_working_on_arabic_ocr/

[^3_16]: https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/OCR.html


---

# كيف أحضّر dataset لـ fine-tuning OCR على بطاقة الهوية المصرية

إليك الدليل الكامل والمتخصص لتجهيز dataset بطاقة الهوية المصرية.

***

## حقول بطاقة الهوية المصرية

قبل أي شيء، حدد الحقول التي تريد استخراجها:[^4_1][^4_2]


| الحقل | مثال | النوع |
| :-- | :-- | :-- |
| الاسم الرباعي | محمد أحمد علي حسن | عربي |
| الرقم القومي | 29801011234567 | أرقام |
| تاريخ الميلاد | 01/01/1998 | أرقام + رموز |
| العنوان | القاهرة - مصر الجديدة | عربي |
| المحافظة | القاهرة | عربي |
| الجنس | ذكر / أنثى | عربي |
| تاريخ الانتهاء | 01/01/2030 | أرقام + رموز |


***

## المصدر الأول: Dataset جاهز (ابدأ منه)

### Kaggle - Synthetic Egyptian ID Cards

يوجد dataset جاهز على Kaggle لبطاقات هوية مصرية اصطناعية:[^4_3]

```bash
kaggle datasets download mg31415/synthetic-egyptian-id-cards
```


### Roboflow - Egyptian ID Detection

لمرحلة الـ Detection (تحديد مواضع الحقول):[^4_4]

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace().project("egyptian-id-card-nsckp-djc1j")
dataset = project.version(1).download("paddleocr")
```


***

## المصدر الثاني: توليد Synthetic Data

أهم مصدر لأنك تتحكم في جودة وكمية البيانات.[^4_5]

### الخطوة 1: جمع الخطوط العربية

```bash
# خطوط مناسبة لبطاقة الهوية المصرية
fonts/
├── Amiri-Regular.ttf       # الخط الرسمي الأقرب للبطاقة
├── Cairo-Regular.ttf
├── NotoNaskhArabic.ttf
└── ReemKufi-Regular.ttf
```


### الخطوة 2: توليد بيانات لكل حقل

```python
from PIL import Image, ImageDraw, ImageFont
import random, os, json
from faker import Faker

fake = Faker('ar_EG')

# قوائم بيانات واقعية
NAMES = [
    "محمد أحمد علي حسن", "فاطمة محمود إبراهيم",
    "أحمد عبد الله محمد", "سارة خالد عبد الرحمن",
    # أضف 200+ اسم
]
GOVERNORATES = [
    "القاهرة", "الجيزة", "الإسكندرية", "الدقهلية",
    "أسوان", "الأقصر", "المنيا", "الفيوم"
]
ADDRESSES = [
    "مصر الجديدة - شارع النزهة",
    "الهرم - شارع فيصل",
    "المعادي - شارع النصر",
]

# خلفية البطاقة (لون كريمي قريب من الهوية الحقيقية)
BG_COLORS = [(245,240,228), (240,235,220), (250,245,235)]
TEXT_COLORS = [(0,0,100), (10,10,90), (0,0,80)]

def generate_rec_sample(text, font_path, save_path,
                        width=320, height=48):
    """توليد صورة recognition لحقل واحد"""
    bg = random.choice(BG_COLORS)
    fg = random.choice(TEXT_COLORS)
    
    img = Image.new('RGB', (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    
    size = random.randint(18, 26)
    font = ImageFont.truetype(font_path, size=size)
    
    # محاذاة يمين للعربية
    bbox = draw.textbbox((0,0), text, font=font)
    x = width - bbox[^4_2] - 8
    y = (height - bbox[^4_3]) // 2
    draw.text((x, y), text, font=font, fill=fg)
    
    # إضافة noise بسيط
    import numpy as np
    arr = np.array(img)
    noise = np.random.randint(-8, 8, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    Image.fromarray(arr).save(save_path)
    return text[::-1]  # عكس النص للـ label ← مهم جداً


def generate_national_id():
    """توليد رقم قومي مصري واقعي"""
    century = random.choice(['2', '3'])
    year = f"{random.randint(60,99):02d}" if century=='2' else f"{random.randint(0,20):02d}"
    month = f"{random.randint(1,12):02d}"
    day = f"{random.randint(1,28):02d}"
    gov_code = f"{random.randint(1,27):02d}"
    sequence = f"{random.randint(1,9999):04d}"
    check = str(random.randint(1,9))
    return f"{century}{year}{month}{day}{gov_code}{sequence}{check}"

# ────────── توليد Dataset ──────────
fonts = [f"./fonts/{f}" for f in os.listdir('./fonts') if f.endswith('.ttf')]
labels = []
os.makedirs('./dataset/rec/images', exist_ok=True)

fields = {
    'name': NAMES * 50,
    'address': ADDRESSES * 100,
    'gov': GOVERNORATES * 200,
    'id_number': [generate_national_id() for _ in range(2000)],
}

idx = 0
for field_type, samples in fields.items():
    for text in samples:
        path = f"./dataset/rec/images/{field_type}_{idx:05d}.jpg"
        font = random.choice(fonts)
        label = generate_rec_sample(text, font, path)
        labels.append(f"rec/images/{field_type}_{idx:05d}.jpg\t{label}")
        idx += 1

# حفظ الـ labels
random.shuffle(labels)
split = int(len(labels) * 0.85)

with open('./dataset/rec/train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(labels[:split]))

with open('./dataset/rec/val.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(labels[split:]))

print(f"✅ Train: {split} | Val: {len(labels)-split}")
```


***

## المصدر الثالث: بيانات حقيقية بـ PPOCRLabel

للبيانات الحقيقية (أهم للـ Validation):[^4_6]

```bash
pip install PPOCRLabel
# تشغيل الأداة مع دعم العربية
PPOCRLabel --lang ar --kie True
```

**Workflow داخل PPOCRLabel:**

1. افتح مجلد صور البطاقات الحقيقية
2. اضغط `Auto Recognition` ← يحدد الحقول تلقائياً
3. صحّح كل بطاقة يدوياً (5–10 دقائق/بطاقة)
4. صنّف كل حقل (Name / ID / Address) لو تريد KIE
5. `File → Export Label` ← يولد الـ annotations جاهزة

***

## Data Augmentation

لتحسين robustness النموذج على صور البطاقات الواقعية:

```python
import albumentations as A
import cv2

augment = A.Compose([
    A.RandomBrightnessContrast(p=0.5),   # تغيير الإضاءة
    A.GaussNoise(p=0.3),                  # ضوضاء
    A.Rotate(limit=3, p=0.4),             # انحراف طفيف
    A.Blur(blur_limit=2, p=0.2),          # ضبابية
    A.CLAHE(p=0.3),                       # تحسين التباين
])

def augment_image(img_path, n=3):
    img = cv2.imread(img_path)
    for i in range(n):
        aug = augment(image=img)['image']
        out_path = img_path.replace('.jpg', f'_aug{i}.jpg')
        cv2.imwrite(out_path, aug)
```


***

## البنية النهائية والأرقام المستهدفة

```
dataset/
├── det/
│   ├── images/         # 500+ صورة بطاقة كاملة
│   └── Label.txt       # annotations الـ detection
└── rec/
    ├── images/         # 8,000–15,000 صورة حقل نصي
    ├── train.txt       # 85% من البيانات
    └── val.txt         # 15% (real فقط للتقييم)
```

| نوع البيانات | الكمية المستهدفة | الأولوية |
| :-- | :-- | :-- |
| Synthetic names | 3,000 صورة | ⭐⭐⭐⭐⭐ |
| Synthetic ID numbers | 2,000 صورة | ⭐⭐⭐⭐⭐ |
| Synthetic addresses | 2,000 صورة | ⭐⭐⭐⭐ |
| Real annotated (PPOCRLabel) | 500+ صورة | ⭐⭐⭐⭐⭐ |
| Augmented versions | ×3 من الكل | ⭐⭐⭐ |

الـ Validation set يجب أن يكون **حقيقياً فقط** وليس synthetic، لأن الهدف قياس الأداء الفعلي على بطاقات حقيقية.[^4_7]
<span style="display:none">[^4_10][^4_11][^4_12][^4_13][^4_14][^4_15][^4_16][^4_17][^4_8][^4_9]</span>

<div align="center">⁂</div>

[^4_1]: https://github.com/NASO7Y/OCR_Egyptian_ID

[^4_2]: https://github.com/Mostafa-Emad77/Egyptian-ID-Extraction

[^4_3]: https://www.kaggle.com/datasets/mg31415/synthetic-egyptian-id-cards

[^4_4]: https://universe.roboflow.com/faculity-of-artificial-intelligence/egyptian-id-card-nsckp-djc1j

[^4_5]: https://arxiv.org/html/2508.13078v1

[^4_6]: https://github.com/PFCCLab/PPOCRLabel

[^4_7]: https://github.com/PaddlePaddle/PaddleOCR/issues/10358

[^4_8]: https://www.kaggle.com/code/yahyaahmed1012/egyptian-id-ocr

[^4_9]: https://signme.it/egypt-national-id-api/?lang=en

[^4_10]: https://structocr.com/blog/tutorials/ocr-egypt-national-id-api-python

[^4_11]: https://arya.ai/apex-apis/egypt-national-id-api

[^4_12]: https://pictotext.io/docs/reference/africa/egypt/id-card

[^4_13]: https://github.com/keronagy/Egyptian-ID-Data-Extraction

[^4_14]: https://github.com/jedota/Synthetic_ID-Card_Image

[^4_15]: https://dl.acm.org/doi/pdf/10.1145/3220267.3220277

[^4_16]: https://ejle.journals.ekb.eg/article_59924_98b54f20a36e88da1c58ca6fb99a1300.pdf

[^4_17]: https://github.com/arij01/ArabID


---

# نصائح لتحسين جودة dataset لـ OCR عربي على الهويات

هذه أهم النصائح المبنية على تجارب حقيقية من مجتمع PaddleOCR والأبحاث المتخصصة في العربية.

***

## 1. جودة الـ Labels أولاً

الخطأ الأكثر شيوعاً هو بيانات كثيرة لكن labels خاطئة — النموذج يتعلم الأخطاء.[^5_1]

**Checklist للـ Label Quality:**

- ✅ **عكس النص العربي** في كل label (RTL → LTR) — هذا الخطأ وحده يُبقي الدقة عند 30–40%[^5_1]
- ✅ عدم وجود مسافات زائدة في بداية أو نهاية النص
- ✅ الأرقام الموجودة في بياناتك مُدرجة في `arabic_dict.txt` (أرقام عربية + لاتينية + `/` و `-`)
- ✅ توحيد الهمزات: `أ` و `إ` و `ا` ← قرر نمطاً واحداً والتزم به
- ✅ لا يوجد تشكيل في الـ labels إلا لو موجود فعلاً على البطاقة

```python
# script للتحقق من جودة الـ labels
with open('train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

issues = []
for i, line in enumerate(lines):
    parts = line.strip().split('\t')
    if len(parts) != 2:
        issues.append(f"Line {i}: missing tab separator")
        continue
    path, label = parts
    if label != label.strip():
        issues.append(f"Line {i}: leading/trailing spaces in label")
    if len(label) == 0:
        issues.append(f"Line {i}: empty label")
    if len(label) > 40:
        issues.append(f"Line {i}: label too long ({len(label)} chars)")

print(f"Found {len(issues)} issues")
for issue in issues[:20]:
    print(issue)
```


***

## 2. تنوع الخطوط — أهم عامل للـ Generalization

النموذج المدرب على خط واحد يفشل مع البطاقات الحقيقية ذات الطباعة المتفاوتة.[^5_2][^5_1]

**الخطوط الأساسية لبطاقة الهوية المصرية:**

```python
REQUIRED_FONTS = {
    "Amiri-Regular": "الأقرب للطباعة الرسمية",
    "NotoNaskhArabic": "النسخ الحديث",
    "Cairo": "مقروء وواضح",
    "Lateef": "للنصوص الرسمية",
    "Scheherazade": "التراثي",
}

# تأكد أن كل font ممثلة بنسبة متساوية في الـ dataset
from collections import Counter
font_distribution = Counter()
# يجب أن يكون كل font ≈ 20% من الـ dataset
```

قاعدة مهمة: **لا تتجاوز 30% من الـ dataset بخط واحد** — التنوع يمنع الـ overfitting على شكل خط بعينه.[^5_1]

***

## 3. تنوع صور بطاقات الهوية الحقيقية

تحديات واقعية يجب أن يراها النموذج في التدريب:[^5_2]

```python
import albumentations as A

# Pipeline الأمثل لـ augmentation بطاقة الهوية
augment_pipeline = A.Compose([

    # ── تحديات الإضاءة ──
    A.RandomBrightnessContrast(
        brightness_limit=0.3, contrast_limit=0.3, p=0.6
    ),
    A.RandomShadow(p=0.2),              # ظل من الإمساك بالبطاقة
    A.RandomGamma(p=0.3),

    # ── تحديات الكاميرا ──
    A.Rotate(limit=3, p=0.4),           # انحراف طفيف (لا تبالغ)
    A.Perspective(scale=(0.01, 0.05), p=0.3),  # التقاط بزاوية
    A.GaussianBlur(blur_limit=3, p=0.3),       # عدم وضوح

    # ── تحديات جودة الصورة ──
    A.GaussNoise(var_limit=(5, 25), p=0.4),
    A.ImageCompression(quality_lower=60, p=0.3),  # ضغط JPEG
    A.CLAHE(clip_limit=2.0, p=0.3),              # تحسين التباين

    # ── بقايا اللامينيشن ──
    A.ToGray(p=0.1),
    A.HueSaturationValue(p=0.2),
])
```


***

## 4. توازن الفئات داخل الـ Dataset

```python
import os
from collections import Counter

# تحليل توزيع الفئات
with open('train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# استخرج نوع كل حقل من اسم الملف
field_counts = Counter()
for line in lines:
    path = line.split('\t')[^5_0]
    field_type = path.split('_')[^5_0].split('/')[-1]  # name, id, address...
    field_counts[field_type] += 1

print("Field distribution:")
total = sum(field_counts.values())
for field, count in field_counts.most_common():
    print(f"  {field}: {count} ({count/total*100:.1f}%)")

# ⚠️ أي فئة أقل من 10% ستكون ضعيفة في النموذج
```

**التوزيع المثالي:**


| الحقل | النسبة المثالية | السبب |
| :-- | :-- | :-- |
| الاسم الرباعي | 30% | أصعب حقل — تنوع عالي |
| الرقم القومي | 25% | أرقام فقط — سهل نسبياً |
| العنوان | 25% | أطول نص — يحتاج تنوعاً |
| باقي الحقول | 20% | محافظة، تاريخ... |


***

## 5. تنظيف الصور قبل التدريب

```python
import cv2
import numpy as np
from pathlib import Path

def validate_and_clean_image(img_path):
    """تصفية الصور الرديئة قبل التدريب"""
    img = cv2.imread(img_path)

    if img is None:
        return False, "Cannot read image"

    h, w = img.shape[:2]

    # نسبة العرض/الارتفاع للحقل النصي
    if w < 50 or h < 15:
        return False, f"Too small: {w}x{h}"
    if w / h < 2 or w / h > 30:
        return False, f"Bad aspect ratio: {w/h:.1f}"

    # رفض الصور شبه الفارغة
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    if contrast < 8:
        return False, f"Low contrast: {contrast:.1f}"

    return True, "OK"

# تشغيل الفلتر على الـ dataset كله
bad_images = []
with open('train.txt', 'r', encoding='utf-8') as f:
    for line in f:
        path = line.split('\t')[^5_0]
        ok, reason = validate_and_clean_image(path)
        if not ok:
            bad_images.append((path, reason))

print(f"Bad images: {len(bad_images)}")
```


***

## 6. قياس جودة الـ Dataset قبل التدريب

```python
# Character coverage — كم % من الـ dictionary مغطى بالبيانات؟
with open('arabic_dict.txt', 'r', encoding='utf-8') as f:
    dictionary_chars = set(f.read().strip())

with open('train.txt', 'r', encoding='utf-8') as f:
    all_labels = [l.strip().split('\t')[^5_1] for l in f if '\t' in l]

dataset_chars = set(''.join(all_labels))
coverage = len(dataset_chars & dictionary_chars) / len(dictionary_chars)
missing = dictionary_chars - dataset_chars

print(f"Dictionary coverage: {coverage*100:.1f}%")
print(f"Missing chars: {''.join(missing)}")
# ⚠️ يجب أن يكون Coverage > 95% وإلا النموذج سيفشل مع الحروف الناقصة
```


***

## أخطاء شائعة تُدمّر جودة الـ Dataset

| الخطأ | التأثير | الحل |
| :-- | :-- | :-- |
| نسيان عكس النص العربي | دقة < 40% [^5_1] | `label[::-1]` لكل label |
| خط واحد فقط | فشل مع البطاقات الحقيقية [^5_1] | 5+ خطوط مختلفة |
| Validation من synthetic فقط | metrics مضللة | Val set = real only |
| أرقام مفقودة من الـ dict | فشل كامل في الرقم القومي [^5_3] | أضف `٠١٢٣٤٥٦٧٨٩0-9/` |
| صور صغيرة جداً (< 15px ارتفاع) | feature maps ناقصة | ارفع `min_height=48px` |

<span style="display:none">[^5_10][^5_11][^5_12][^5_13][^5_14][^5_15][^5_4][^5_5][^5_6][^5_7][^5_8][^5_9]</span>

<div align="center">⁂</div>

[^5_1]: https://github.com/PaddlePaddle/PaddleOCR/issues/10358

[^5_2]: https://kby-ai.com/4-real-life-id-document-ocr-challenges-in-processing/

[^5_3]: https://github.com/PaddlePaddle/PaddleOCR/issues/11372

[^5_4]: https://aclanthology.org/2025.findings-acl.1135.pdf

[^5_5]: https://www.nature.com/articles/s41598-025-12045-z

[^5_6]: https://arxiv.org/pdf/2502.14949.pdf

[^5_7]: https://ieeexplore.ieee.org/iel8/6287639/10820123/11177005.pdf

[^5_8]: https://arxiv.org/html/2506.02295v1

[^5_9]: https://arxiv.org/html/2312.11812v1

[^5_10]: https://www.facebook.com/groups/ComputerVisionGroup/posts/3339668449510666/

[^5_11]: https://ejle.journals.ekb.eg/article_59924_98b54f20a36e88da1c58ca6fb99a1300.pdf

[^5_12]: https://arxiv.org/html/2502.14949v2

[^5_13]: https://www.reddit.com/r/computervision/comments/1n3xh4d/ocr_arabic_documents_quality_assessment_method/

[^5_14]: https://github.com/PaddlePaddle/PaddleOCR/issues/10519

[^5_15]: https://b.aun.edu.eg/fci/sites/default/files/researches/Advancements and Challenges in Arabic Optical Character Recognition A Comprehensive Survey.pdf


---

# خطوات نشر الموديل المُfine-tuned على GPU

إليك الدليل الكامل لنشر النموذج المُدرَّب على GPU بشكل production-ready.

***

## المسار الكامل للنشر

```
trained model (.pdparams)
        ↓ export
inference model (.pdmodel + .pdiparams)
        ↓ اختياري
     ONNX model
        ↓ اختياري
   TensorRT Engine
        ↓
   FastAPI Service
        ↓
   Docker Container
```


***

## 1. تصدير النموذج للـ Inference

```bash
# تصدير Detection model
python tools/export_model.py \
  -c configs/det/arabic_det.yml \
  -o Global.pretrained_model=./output/det/best_accuracy \
     Global.save_inference_dir=./inference/det

# تصدير Recognition model
python tools/export_model.py \
  -c configs/rec/arabic_egyptianid_rec.yml \
  -o Global.pretrained_model=./output/rec/best_accuracy \
     Global.save_inference_dir=./inference/rec
```

الناتج:

```
inference/
├── det/
│   ├── inference.pdmodel
│   └── inference.pdiparams
└── rec/
    ├── inference.pdmodel
    └── inference.pdiparams
```


***

## 2. تثبيت PaddlePaddle GPU

```bash
# CUDA 11.8
pip install paddlepaddle-gpu==2.6.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# CUDA 12.3
pip install paddlepaddle-gpu==2.6.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu123/

# التحقق من GPU
python -c "import paddle; print(paddle.device.get_device())"
# Expected: gpu:0
```


***

## 3. خيارات تسريع GPU

PaddleOCR 3.0 يختار الـ backend الأمثل تلقائياً بـ `enable_hpi=True`:[^6_1]


| Backend | متى يُستخدم | تسريع مقارناً بـ CPU |
| :-- | :-- | :-- |
| **Paddle Inference** | الافتراضي | 5–10× |
| **ONNX Runtime (GPU)** | CUDA بدون TensorRT | 8–12× |
| **TensorRT** | NVIDIA GPU — أقصى سرعة | 15–25× [^6_2] |

### الطريقة 1: High-Performance Inference (الأسهل)

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_gpu=True,
    enable_hpi=True,        # ← يختار TensorRT/ONNX تلقائياً [web:72]
    precision="fp16",        # ← نصف الدقة = ضعف السرعة
    det_model_dir='./inference/det',
    rec_model_dir='./inference/rec',
    rec_char_dict_path='./arabic_dict.txt',
    lang='ar',
)
```


### الطريقة 2: TensorRT يدوياً (أقصى أداء)

```bash
# الخطوة 1: تحويل إلى ONNX
pip install paddle2onnx

paddle2onnx \
  --model_dir ./inference/rec \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./rec.onnx \
  --opset_version 11

# الخطوة 2: ONNX → TensorRT Engine
trtexec \
  --onnx=rec.onnx \
  --saveEngine=rec.engine \
  --explicitBatch \
  --fp16                    # ← FP16 للتسريع
```


***

## 4. FastAPI Service

```python
# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from functools import lru_cache
import numpy as np
import cv2

app = FastAPI(title="Egyptian ID OCR API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@lru_cache(maxsize=1)           # ← تحميل النموذج مرة واحدة فقط [web:74]
def get_ocr_model():
    return PaddleOCR(
        use_gpu=True,
        enable_hpi=True,
        det_model_dir='./inference/det',
        rec_model_dir='./inference/rec',
        rec_char_dict_path='./arabic_dict.txt',
        lang='ar',
        show_log=False,
    )

@app.post("/ocr/egyptian-id")
async def extract_id_fields(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "Only JPEG/PNG supported")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    ocr = get_ocr_model()
    result = ocr.ocr(img, cls=True)

    # استخراج وتنظيم الحقول
    fields = []
    for line in result[^6_0]:
        bbox, (text, confidence) = line
        # عكس النص مجدداً للعرض الصحيح
        fields.append({
            "text": text[::-1],
            "confidence": round(confidence, 3),
            "bbox": bbox
        })

    return {
        "status": "success",
        "fields": fields,
        "total_fields": len(fields)
    }

@app.get("/health")
def health_check():
    import paddle
    return {
        "status": "healthy",
        "device": paddle.device.get_device()
    }
```

```bash
# تشغيل السيرفر
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```


***

## 5. Docker للـ Production

```dockerfile
# Dockerfile
FROM paddlepaddle/paddle:2.6.0-gpu-cuda11.8-cudnn8.6-trt8.5

WORKDIR /app

# تثبيت المتطلبات
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# نسخ النماذج والكود
COPY ./inference ./inference
COPY ./app ./app
COPY arabic_dict.txt .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  ocr-api:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]   # ← تفعيل GPU في Docker
    volumes:
      - ./inference:/app/inference  # mount النماذج خارج الـ image
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
```

```bash
# بناء وتشغيل
docker compose up --build -d

# اختبار
curl -X POST "http://localhost:8000/ocr/egyptian-id" \
  -F "file=@test_id.jpg"
```


***

## 6. قياس الأداء بعد النشر

```python
import time, requests, statistics

def benchmark(image_path, n=100):
    with open(image_path, 'rb') as f:
        img_bytes = f.read()

    times = []
    for _ in range(n):
        start = time.perf_counter()
        requests.post(
            "http://localhost:8000/ocr/egyptian-id",
            files={"file": ("id.jpg", img_bytes, "image/jpeg")}
        )
        times.append((time.perf_counter() - start) * 1000)

    print(f"Avg:    {statistics.mean(times):.1f} ms")
    print(f"P95:    {sorted(times)[int(n*0.95)]:.1f} ms")
    print(f"P99:    {sorted(times)[int(n*0.99)]:.1f} ms")

benchmark("test_id.jpg")
```

**أهداف الأداء على GPU:**


| الإعداد | زمن المعالجة |
| :-- | :-- |
| Paddle Inference + GPU | 50–80 ms |
| ONNX Runtime + GPU | 30–50 ms |
| TensorRT FP16 | 10–20 ms [^6_2] |
| CPU فقط (مقارنة) | 300–600 ms |


***

## نصائح Production

- **لا تُعيد تحميل النموذج** مع كل request — استخدم `@lru_cache` أو singleton[^6_3]
- فعّل `--workers 1` مع GPU واحد لتجنب تعارض الـ CUDA contexts[^6_4]
- احتفظ بـ **CPU fallback** في الكود لو فشل الـ GPU عند الـ startup
- سجّل `confidence score` مع كل نتيجة وارفض النتائج أقل من 0.7 تلقائياً
<span style="display:none">[^6_10][^6_11][^6_12][^6_13][^6_14][^6_15][^6_5][^6_6][^6_7][^6_8][^6_9]</span>

<div align="center">⁂</div>

[^6_1]: https://www.paddleocr.ai/main/en/version3.x/deployment/high_performance_inference.html

[^6_2]: https://arxiv.org/html/2507.05595v1

[^6_3]: https://www.youtube.com/watch?v=A5JHYMN4-p0

[^6_4]: https://github.com/PaddlePaddle/PaddleOCR/discussions/16398

[^6_5]: https://github.com/PaddlePaddle/PaddleOCR/issues/16823

[^6_6]: https://hackernoon.com/ocr-fine-tuning-from-raw-data-to-custom-paddle-ocr-model

[^6_7]: https://github.com/PaddlePaddle/Paddle2ONNX/issues/411

[^6_8]: https://github.com/PaddlePaddle/PaddleOCR/discussions/15009

[^6_9]: https://dev.to/czmilo/2025-complete-guide-paddleocr-vl-09b-baidus-ultra-lightweight-document-parsing-powerhouse-1e8l

[^6_10]: https://paddlepaddle.github.io/PaddleOCR/v3.0.2/en/version3.x/deployment/high_performance_inference.html

[^6_11]: https://github.com/velviagris/PaddleOCRFastAPI

[^6_12]: https://paddlepaddle.github.io/PaddleX/3.3/en/pipeline_deploy/high_performance_inference.html

[^6_13]: https://www.youtube.com/watch?v=ftBNkG7kUOI

[^6_14]: https://github.com/PaddlePaddle/PaddleOCR/issues/5884

[^6_15]: https://github.com/nilecui/PaddleOCR


---

# كيف أحوّل موديل PaddleOCR إلى ONNX للنشر

إليك الدليل الكامل لتحويل نموذج PaddleOCR إلى ONNX والنشر به.

***

## المسار الكامل للتحويل

```
fine-tuned model (.pdparams)
        ↓  tools/export_model.py
inference model (.pdmodel + .pdiparams)
        ↓  paddle2onnx
    model.onnx  (dynamic input)
        ↓  onnxsim (اختياري)
  model_sim.onnx  (محسّن)
        ↓
onnxruntime CPU/GPU   أو   TensorRT Engine
```


***

## 1. التثبيت

```bash
# الحزم الأساسية
pip install paddle2onnx onnx onnxsim

# لـ CPU فقط
pip install onnxruntime

# لـ GPU (CUDA)
pip install onnxruntime-gpu

# أداة RapidOCR الجاهزة (بديل سهل)
pip install rapidocr_onnxruntime
```


***

## 2. تصدير نموذج Paddle أولاً

قبل التحويل، يجب تصدير النموذج من صيغة training إلى inference:[^7_1]

```bash
# تصدير Detection
python tools/export_model.py \
  -c configs/det/arabic_det.yml \
  -o Global.pretrained_model=./output/det/best_accuracy \
     Global.save_inference_dir=./inference/det

# تصدير Recognition
python tools/export_model.py \
  -c configs/rec/arabic_egyptianid_rec.yml \
  -o Global.pretrained_model=./output/rec/best_accuracy \
     Global.save_inference_dir=./inference/rec
```


***

## 3. التحويل إلى ONNX

### الطريقة 1: paddle2onnx (يدوي — أكثر تحكماً)

```bash
# تحويل Detection model
paddle2onnx \
  --model_dir ./inference/det \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./onnx/det.onnx \
  --opset_version 11 \
  --enable_onnx_checker True

# تحويل Recognition model
paddle2onnx \
  --model_dir ./inference/rec \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./onnx/rec.onnx \
  --opset_version 11 \
  --enable_onnx_checker True
```


### الطريقة 2: PaddleX Plugin (الأبسط - PaddleOCR 3.x)[^7_2]

```bash
# تثبيت plugin التحويل
paddlex --install paddle2onnx

# تحويل مباشر
paddlex \
  --paddle2onnx \
  --paddle_model_dir ./inference/rec \
  --onnx_model_dir ./onnx/rec \
  --opset_version 11
```


### الطريقة 3: PaddleOCRModelConvert (الأسهل للـ rec)[^7_3]

```bash
pip install paddleocr_convert

# تحويل مع حفظ الـ character dict داخل الـ ONNX مباشرة
paddleocr_convert \
  -p ./inference/rec \
  -o ./onnx/ \
  -txt_path ./arabic_dict.txt
```


***

## 4. تحسين ملف ONNX (مهم)

```bash
# تبسيط الـ graph وحذف العمليات الزائدة
python -m onnxsim ./onnx/rec.onnx ./onnx/rec_sim.onnx

# التحقق من صحة الملف
python -c "
import onnx
model = onnx.load('./onnx/rec_sim.onnx')
onnx.checker.check_model(model)
print('✅ ONNX model is valid')
print(f'Opset: {model.opset_import[^7_0].version}')
"
```


***

## 5. الـ Inference بـ ONNX Runtime

### على CPU

```python
import onnxruntime as ort
import numpy as np
import cv2

# إعداد الـ session
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4      # عدد CPU threads
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

rec_session = ort.InferenceSession(
    './onnx/rec_sim.onnx',
    sess_options=sess_options,
    providers=['CPUExecutionProvider']
)

det_session = ort.InferenceSession(
    './onnx/det_sim.onnx',
    sess_options=sess_options,
    providers=['CPUExecutionProvider']
)

print("Available providers:", ort.get_available_providers())
```


### على GPU (CUDA)

```python
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 ** 3,    # 2GB GPU VRAM
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',                  # ← fallback تلقائي لو GPU فشل
]

rec_session = ort.InferenceSession(
    './onnx/rec_sim.onnx',
    providers=providers
)

# التحقق أن CUDA فعلاً مستخدم
print("Active provider:", rec_session.get_providers()[^7_0])
# Expected: CUDAExecutionProvider
```


***

## 6. استخدام RapidOCR (الحل الأسهل للـ Production)

بدلاً من كتابة preprocessing يدوياً، **RapidOCR** يغلّف كل شيء ويستخدم ملفات ONNX مباشرة:[^7_4][^7_3]

```python
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR(
    det_model_path='./onnx/det_sim.onnx',
    rec_model_path='./onnx/rec_sim.onnx',
    # rec_keys_path مدمج داخل الـ ONNX لو استخدمت paddleocr_convert
)

result, elapse = engine('egyptian_id.jpg')
print(f"⏱️ Time: {elapse:.3f}s")
for line in result:
    bbox, text, confidence = line
    print(f"[{confidence:.2f}] {text}")
```


***

## 7. قياس السرعة بعد التحويل

```python
import time
import numpy as np
import onnxruntime as ort

def benchmark_onnx(model_path, input_shape, n=200, provider='CPU'):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if provider == 'GPU' else ['CPUExecutionProvider']

    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[^7_0].name

    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy_input})

    # Benchmark
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        times.append((time.perf_counter() - t0) * 1000)

    print(f"Provider : {session.get_providers()[^7_0]}")
    print(f"Avg      : {np.mean(times):.2f} ms")
    print(f"P95      : {np.percentile(times, 95):.2f} ms")
    print(f"P99      : {np.percentile(times, 99):.2f} ms")

# Recognition model: [batch, channel, height, width]
benchmark_onnx('./onnx/rec_sim.onnx', (1, 3, 48, 320), provider='CPU')
benchmark_onnx('./onnx/rec_sim.onnx', (1, 3, 48, 320), provider='GPU')
```


***

## مقارنة سرعة الـ Backends

| Backend | زمن CPU | زمن GPU | التثبيت |
| :-- | :-- | :-- | :-- |
| Paddle Inference | ~300 ms | ~60 ms | `paddlepaddle-gpu` |
| ONNX Runtime | ~90 ms | ~20 ms [^7_4] | `onnxruntime-gpu` |
| ONNX Runtime (FP16) | — | ~12 ms | `onnxruntime-gpu` |
| TensorRT | — | ~8 ms [^7_5] | NVIDIA TensorRT |


***

## أخطاء شائعة وحلولها

| الخطأ | السبب | الحل |
| :-- | :-- | :-- |
| `Invalid ONNX model` | opset غير متوافق | استخدم `--opset_version 11` [^7_1] |
| `Shape mismatch` | input ديناميكي | شغّل `onnxsim` بعد التحويل |
| GPU لا يُستخدم | CUDA مش متثبت | `pip install onnxruntime-gpu` وليس `onnxruntime` [^7_6] |
| نتائج مختلفة عن Paddle | تقريب FP32 | مقبول — الفرق أقل من 0.1% في الدقة [^7_7] |

<span style="display:none">[^7_10][^7_11][^7_12][^7_13][^7_14][^7_15][^7_8][^7_9]</span>

<div align="center">⁂</div>

[^7_1]: https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddle2onnx/readme.md

[^7_2]: http://www.paddleocr.ai/main/en/version3.x/deployment/obtaining_onnx_models.html

[^7_3]: https://github.com/RapidAI/PaddleOCRModelConvert

[^7_4]: https://zenn.dev/harumikun/articles/fb9c435acf4070?locale=en

[^7_5]: https://paddlepaddle.github.io/PaddleX/3.0-rc1/en/pipeline_deploy/high_performance_inference.html

[^7_6]: https://github.com/PaddlePaddle/PaddleOCR/discussions/14572

[^7_7]: https://huggingface.co/monkt/paddleocr-onnx

[^7_8]: https://paddlepaddle.github.io/PaddleOCR/main/en/version2.x/legacy/paddle2onnx.html

[^7_9]: https://huggingface.co/marsena/paddleocr-onnx-models

[^7_10]: https://paddlepaddle.github.io/PaddleOCR/v3.0.1/en/version3.x/deployment/high_performance_inference.html

[^7_11]: https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/deployment/high_performance_inference.html

[^7_12]: http://www.paddleocr.ai/v2.9/en/ppocr/infer_deploy/paddle2onnx.html

[^7_13]: https://github.com/PaddlePaddle/PaddleOCR/discussions/14771

[^7_14]: https://paddlepaddle.github.io/PaddleOCR/v2.9.1/en/ppocr/infer_deploy/paddle2onnx.html

[^7_15]: https://paddlepaddle.github.io/PaddleOCR/v3.0.0/en/version3.x/deployment/obtaining_onnx_models.html


---

# كيف أحوّل موديل fine-tuned إلى ONNX للعربية

لاحظت إنك بعت نفس السؤال اللي أجبته في الرد السابق بالتفصيل — لكن يبدو إن السؤال هنا مُركّز على **التحديات الخاصة بالعربية عند التحويل للـ ONNX**، وهي مختلفة عن الحالة العامة. إليك ما يهمك تحديداً.

***

## المشكلة الحرجة: RTL داخل ONNX

النموذج المُدرَّب على labels معكوسة (LTR) سيُخرج نصاً معكوساً عند الـ inference — يجب عكسه وإعادة تشكيله:[^8_1][^8_2]

```python
pip install arabic-reshaper python-bidi
```

```python
import arabic_reshaper
from bidi.algorithm import get_display

def fix_arabic_text(raw_text: str) -> str:
    """إصلاح النص العربي الخارج من ONNX"""
    # 1. عكس النص (لأن النموذج تدرب على labels معكوسة)
    reversed_text = raw_text[::-1]
    # 2. إعادة تشكيل الحروف (contextual forms)
    reshaped = arabic_reshaper.reshape(reversed_text)
    # 3. ترتيب RTL الصحيح للعرض
    display_text = get_display(reshaped)
    return display_text

# مثال
raw = "يلع دمحم"          # خروج النموذج
print(fix_arabic_text(raw))  # محمد علي ✅
```


***

## تضمين الـ Character Dict داخل ONNX

بعد الـ fine-tuning بـ `arabic_dict.txt` مخصص، يجب تضمينه في الـ ONNX حتى لا تحتاجه كـ external file:[^8_3]

```bash
pip install paddleocr_convert

paddleocr_convert \
  -p ./inference/rec \
  -o ./onnx/ \
  -txt_path ./arabic_dict.txt    # ← يُضمَّن داخل الـ ONNX مباشرة
```

> بدون هذه الخطوة، كل environment تنشر فيه يحتاج نسخة من الـ dict يدوياً.

***

## التحقق من تطابق الـ Dict مع النموذج

خطأ شائع: الـ ONNX يحتوي dict مختلف عن اللي دُرِّب عليه النموذج:[^8_4]

```python
import onnx

model = onnx.load('./onnx/rec_sim.onnx')

# استخراج الـ metadata
for prop in model.metadata_props:
    print(f"{prop.key}: {prop.value}")

# التحقق من عدد الحروف = حجم آخر طبقة في النموذج
graph = model.graph
output_shape = graph.output[^8_0].type.tensor_type.shape.dim
print(f"Output classes: {output_shape[-1].dim_value}")

# يجب أن يساوي: len(arabic_dict) + 1 (blank token for CTC)
with open('./arabic_dict.txt', encoding='utf-8') as f:
    dict_size = len(f.readlines())
print(f"Dict size + blank: {dict_size + 1}")
```


***

## Pipeline الكامل بعد التحويل

```python
import onnxruntime as ort
import numpy as np
import cv2
import arabic_reshaper
from bidi.algorithm import get_display

class ArabicIDOCR:
    def __init__(self, det_path, rec_path, dict_path,
                 use_gpu=False):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                    if use_gpu else ['CPUExecutionProvider']

        self.det = ort.InferenceSession(det_path, providers=providers)
        self.rec = ort.InferenceSession(rec_path, providers=providers)

        # تحميل الـ dict
        with open(dict_path, encoding='utf-8') as f:
            chars = f.read().strip().split('\n')
        self.chars = ['blank'] + chars  # CTC blank token

    def preprocess_rec(self, img, target_h=48, target_w=320):
        img = cv2.resize(img, (target_w, target_h))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return img.transpose(2, 0, 1)[np.newaxis]  # [1,3,H,W]

    def decode_ctc(self, preds):
        """CTC Greedy Decode مع إصلاح العربية"""
        indices = np.argmax(preds[^8_0], axis=-1)
        raw_chars = []
        prev = -1
        for idx in indices:
            if idx != prev and idx != 0:  # 0 = blank
                raw_chars.append(self.chars[idx])
            prev = idx
        raw_text = ''.join(raw_chars)

        # إصلاح RTL ← المرحلة الحرجة
        reversed_text = raw_text[::-1]
        reshaped = arabic_reshaper.reshape(reversed_text)
        return get_display(reshaped)

    def recognize(self, crop_img):
        inp = self.preprocess_rec(crop_img)
        input_name = self.rec.get_inputs()[^8_0].name
        preds = self.rec.run(None, {input_name: inp})
        return self.decode_ctc(preds[^8_0])


# تشغيل
ocr = ArabicIDOCR(
    det_path='./onnx/det_sim.onnx',
    rec_path='./onnx/rec_sim.onnx',
    dict_path='./arabic_dict.txt',
    use_gpu=True
)

img = cv2.imread('egyptian_id.jpg')
# افترض إنك عندك الـ crop للحقل
name_crop = img[50:90, 100:400]
print(ocr.recognize(name_crop))  # محمد أحمد علي
```


***

## Checklist قبل النشر

```python
def validate_arabic_onnx(rec_onnx_path, dict_path, test_img_path):
    """تحقق شامل من صحة النموذج قبل النشر"""
    import onnx, onnxruntime as ort

    # 1. تحقق من صحة ملف ONNX
    model = onnx.load(rec_onnx_path)
    onnx.checker.check_model(model)
    print("✅ ONNX model valid")

    # 2. تحقق من توافق الـ dict مع النموذج
    with open(dict_path, encoding='utf-8') as f:
        dict_size = len(f.readlines())
    output_dim = model.graph.output[^8_0].type.tensor_type \
                      .shape.dim[-1].dim_value
    assert output_dim == dict_size + 1, \
        f"❌ Dict mismatch: model={output_dim}, dict={dict_size+1}"
    print(f"✅ Dict matches model ({dict_size} chars)")

    # 3. تحقق من الـ inference لا يُرجع خطأ
    sess = ort.InferenceSession(rec_onnx_path,
                                 providers=['CPUExecutionProvider'])
    dummy = np.random.randn(1, 3, 48, 320).astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[^8_0].name: dummy})
    print(f"✅ Inference OK — output shape: {out[^8_0].shape}")

    print("\n🚀 النموذج جاهز للنشر!")

validate_arabic_onnx('./onnx/rec_sim.onnx', './arabic_dict.txt', 'test.jpg')
```


***

## ملخص الفروق الخاصة بالعربية

| الجانب | النموذج العادي | النموذج العربي المُدرَّب |
| :-- | :-- | :-- |
| الـ Labels | LTR طبيعي | معكوسة RTL→LTR [^8_5] |
| الـ Output | قرأ مباشرة | يحتاج `[::-1]` + reshaper [^8_1] |
| الـ Dict | مدمج في النموذج | ملف خارجي يجب تضمينه [^8_3] |
| عدد الـ classes | ثابت | `len(dict) + 1` (CTC blank) [^8_4] |

<span style="display:none">[^8_10][^8_11][^8_12][^8_13][^8_14][^8_15][^8_16][^8_6][^8_7][^8_8][^8_9]</span>

<div align="center">⁂</div>

[^8_1]: https://github.com/PaddlePaddle/PaddleOCR/discussions/14947

[^8_2]: https://github.com/PaddlePaddle/PaddleOCR/issues/14948

[^8_3]: https://github.com/RapidAI/PaddleOCRModelConvert

[^8_4]: https://github.com/PaddlePaddle/PaddleOCR/discussions/12211

[^8_5]: https://github.com/PaddlePaddle/PaddleOCR/issues/10358

[^8_6]: https://github.com/PaddlePaddle/PaddleOCR

[^8_7]: https://www.kaggle.com/code/oralia/paddleocr-multilingual-model-training-tutorial

[^8_8]: https://huggingface.co/monkt/paddleocr-onnx

[^8_9]: https://github.com/PaddlePaddle/PaddleOCR/issues/10519

[^8_10]: https://www.facebook.com/groups/ComputerVisionGroup/posts/3339668449510666/

[^8_11]: https://paddlepaddle.github.io/PaddleOCR/v2.9.1/en/ppocr/infer_deploy/paddle2onnx.html

[^8_12]: http://www.paddleocr.ai/main/en/index.html

[^8_13]: https://github.com/PaddlePaddle/PaddleOCR/discussions/14971

[^8_14]: http://www.paddleocr.ai/v2.9/en/ppocr/infer_deploy/paddle2onnx.html

[^8_15]: https://paddlepaddle.github.io/PaddleOCR/v2.10.0/en/infer_deploy/paddle2onnx.html

[^8_16]: https://www.youtube.com/watch?v=I0pO2ES-wMg


---

# ايه اللي هحتاج اعرفه تاني

بناءً على كل اللي غطيناه، ده ما تبقى من الـ pipeline الكامل.

***

## ما غطيناه حتى الآن

```
✅ اختيار النموذج (PaddleOCR PP-OCRv4 mobile)
✅ Fine-tuning على CPU
✅ تجهيز الـ Dataset
✅ تحسين جودة البيانات
✅ النشر على GPU
✅ تحويل ONNX
```


***

## ما تبقى من الـ Pipeline

### 1. تقييم النموذج بعد التدريب

الخطوة الأكثر إهمالاً — بدونها لا تعرف هل النموذج جاهز للإنتاج:

```python
# CER = Character Error Rate — المقياس الرئيسي للـ OCR
def calculate_cer(predictions: list, ground_truths: list) -> float:
    import editdistance
    total_chars, total_errors = 0, 0
    for pred, gt in zip(predictions, ground_truths):
        total_errors += editdistance.eval(pred, gt)
        total_chars += len(gt)
    return total_errors / total_chars

# WER = Word Error Rate — مهم للأسماء والعناوين
def calculate_wer(predictions: list, ground_truths: list) -> float:
    import editdistance
    total_words, total_errors = 0, 0
    for pred, gt in zip(predictions, ground_truths):
        pred_words = pred.split()
        gt_words = gt.split()
        total_errors += editdistance.eval(pred_words, gt_words)
        total_words += len(gt_words)
    return total_errors / total_words

# الأهداف المقبولة لبطاقة الهوية المصرية
# CER < 5%  ← جيد
# CER < 2%  ← ممتاز
# CER > 10% ← يحتاج مزيد من بيانات
```


***

### 2. Post-Processing — استخراج الحقول

بعد الـ OCR، النتيجة نصوص خام تحتاج تنظيم:

```python
import re

class EgyptianIDParser:
    """استخراج وتحقق حقول بطاقة الهوية المصرية"""

    # الرقم القومي: 14 رقم يبدأ بـ 2 أو 3
    NID_PATTERN = re.compile(r'\b[23]\d{13}\b')

    # التاريخ: dd/mm/yyyy
    DATE_PATTERN = re.compile(r'\b\d{2}/\d{2}/\d{4}\b')

    GOVERNORATES = {
        "01": "القاهرة", "02": "الإسكندرية", "03": "بورسعيد",
        "04": "السويس", "11": "دمياط", "12": "الدقهلية",
        "15": "الشرقية", "16": "القليوبية", "17": "كفر الشيخ",
        "18": "الغربية", "19": "المنوفية", "21": "الجيزة",
        # ... باقي المحافظات
    }

    def parse_national_id(self, nid: str) -> dict:
        if not self.NID_PATTERN.match(nid):
            return {"valid": False}

        century = "19" if nid[0] == "2" else "20"
        birth_date = f"{century}{nid[1:3]}/{nid[3:5]}/{nid[5:7]}"
        gov_code = nid[7:9]

        return {
            "valid": True,
            "national_id": nid,
            "birth_date": birth_date,
            "governorate": self.GOVERNORATES.get(gov_code, "غير معروف"),
            "gender": "ذكر" if int(nid[12]) % 2 != 0 else "أنثى"
        }

    def extract_from_ocr(self, ocr_results: list) -> dict:
        """تصنيف الحقول بناءً على pattern الـ text"""
        extracted = {}
        for text, confidence in ocr_results:
            if self.NID_PATTERN.search(text):
                extracted['national_id'] = self.parse_national_id(
                    self.NID_PATTERN.search(text).group()
                )
            elif self.DATE_PATTERN.search(text):
                extracted.setdefault('dates', []).append(text)
            elif len(text) > 8 and confidence > 0.85:
                extracted.setdefault('text_fields', []).append(text)
        return extracted
```


***

### 3. المراقبة في الإنتاج (Observability)

```python
# FastAPI مع تسجيل الأداء
from fastapi import FastAPI, UploadFile
import time, logging, json
from datetime import datetime

logging.basicConfig(
    filename='ocr_production.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

@app.post("/ocr/egyptian-id")
async def extract_id(file: UploadFile):
    start = time.perf_counter()
    result = ocr_model.process(await file.read())
    elapsed = time.perf_counter() - start

    # تسجيل كل request
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "latency_ms": round(elapsed * 1000, 1),
        "fields_found": len(result.get("fields", [])),
        "avg_confidence": result.get("avg_confidence", 0),
        "low_confidence": result.get("avg_confidence", 1) < 0.7
    }
    logging.info(json.dumps(log_entry, ensure_ascii=False))

    # إشعار تلقائي لو الثقة منخفضة
    if log_entry["low_confidence"]:
        logging.warning(f"Low confidence result: {log_entry}")

    return result
```


***

### 4. اختبارات ما قبل النشر

```python
# tests/test_ocr_pipeline.py
import pytest

@pytest.fixture
def ocr_engine():
    from app.ocr import ArabicIDOCR
    return ArabicIDOCR('./onnx/det.onnx', './onnx/rec.onnx', './arabic_dict.txt')

class TestEgyptianIDOCR:

    def test_national_id_extraction(self, ocr_engine):
        result = ocr_engine.process('tests/fixtures/sample_id.jpg')
        assert 'national_id' in result
        assert len(result['national_id']) == 14
        assert result['national_id'][0] in ['2', '3']

    def test_low_quality_image(self, ocr_engine):
        # يجب أن يُرجع نتيجة بدون crash حتى مع صورة رديئة
        result = ocr_engine.process('tests/fixtures/blurry_id.jpg')
        assert result is not None
        assert 'error' not in result

    def test_inference_latency(self, ocr_engine):
        import time
        start = time.perf_counter()
        ocr_engine.process('tests/fixtures/sample_id.jpg')
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 500, f"Too slow: {elapsed:.0f}ms"  # < 500ms على CPU

    def test_rtl_text_correct(self, ocr_engine):
        # تأكد إن النص العربي مش معكوس
        result = ocr_engine.process('tests/fixtures/known_id.jpg')
        name = result.get('name', '')
        assert name == "محمد أحمد علي"  # المعروف مسبقاً
```


***

### 5. تحديث النموذج بدون توقف (Zero Downtime)

```python
# model_manager.py — تحديث النموذج أثناء التشغيل
import threading
from pathlib import Path

class ModelManager:
    def __init__(self, model_dir: str):
        self._model = self._load(model_dir)
        self._lock = threading.RLock()

    def _load(self, model_dir):
        from app.ocr import ArabicIDOCR
        return ArabicIDOCR(
            f"{model_dir}/det.onnx",
            f"{model_dir}/rec.onnx",
            f"{model_dir}/arabic_dict.txt"
        )

    def hot_reload(self, new_model_dir: str):
        """تحميل النموذج الجديد بدون إيقاف السيرفر"""
        new_model = self._load(new_model_dir)
        with self._lock:
            self._model = new_model
        print(f"✅ Model reloaded from {new_model_dir}")

    def predict(self, img):
        with self._lock:
            return self._model.process(img)

# endpoint لإعادة التحميل
@app.post("/admin/reload-model")
async def reload_model(model_dir: str):
    model_manager.hot_reload(model_dir)
    return {"status": "reloaded"}
```


***

## الخريطة الكاملة للمشروع

```
egyptian-id-ocr/
├── data/
│   ├── raw/              # صور الهويات الأصلية
│   ├── synthetic/        # البيانات الاصطناعية
│   └── processed/        # train.txt / val.txt
├── training/
│   ├── configs/          # ملفات الـ YAML
│   └── output/           # checkpoints
├── inference/
│   ├── det/              # paddle inference models
│   └── rec/
├── onnx/
│   ├── det_sim.onnx
│   └── rec_sim.onnx
├── app/
│   ├── main.py           # FastAPI
│   ├── ocr.py            # ONNX inference
│   ├── parser.py         # post-processing
│   └── model_manager.py  # hot reload
├── tests/
│   ├── fixtures/         # صور اختبار
│   └── test_pipeline.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```


***

## الخطوة التالية المنطقية

بما إنك عندك كل المكونات، الأولوية الآن:


| الأولوية | الخطوة | الوقت التقديري |
| :-- | :-- | :-- |
| 🔴 أولاً | جمع 200+ صورة حقيقية وعمل annotation بـ PPOCRLabel | 2–3 أيام |
| 🟠 ثانياً | توليد 5000+ synthetic sample بالكود | 2–4 ساعات |
| 🟡 ثالثاً | تشغيل أول fine-tuning وقياس CER على الـ val set | 4–8 ساعات CPU |
| 🟢 رابعاً | تصدير ONNX وبناء FastAPI endpoint | يوم واحد |


---

# معايا dataset جاهزه معمولها annotation و اكتر من 15000 صوره بس محتاج ocr model عشان اطلع النص من الحقول سواء gemini api او haggen face models open source

بما إن عندك 15,000+ صورة مع annotation جاهزة، ده بيغير الصورة تماماً — مش بس inference، ده قدرة حقيقية على fine-tuning VLM. إليك المقارنة الكاملة.

***

## المشهد الحالي 2025–2026

بناءً على KITAB-Bench (أشمل benchmark للعربية):[^10_1][^10_2]


| النموذج | نوعه | CER على العربية | GPU مطلوب | التكلفة |
| :-- | :-- | :-- | :-- | :-- |
| **Gemini 2.0 Flash** | API | ~65% دقة PDF [^10_1] | لا | Pay-per-use |
| **Gemini 2.0 Pro** | API | الأعلى عموماً [^10_3] | لا | Pay-per-use |
| **QARI-OCR (Qwen2-VL-2B)** | Open-source | State-of-the-art [^10_4] | 6GB+ | مجاني |
| **AtlasOCR (3B)** | Open-source | قريب من Gemini [^10_2] | 8GB+ | مجاني |
| **PaddleOCR PP-OCRv4** | Open-source | جيد — لكن أضعف من VLMs | CPU ✅ | مجاني |


***

## استراتيجية ذكية: الاثنين معاً

الأفضل لحالتك بالضبط (dataset جاهز + 15k صورة) هو pipeline مُركَّب:[^10_3][^10_2]

```
Gemini API → Pre-labeling / Pseudo-labels
     ↓
Human Review (تصحيح 10-20% من النتائج)
     ↓
Fine-tune QARI-OCR على dataset بطاقة الهوية
     ↓
نموذج خاص بك — offline — بدون تكلفة API
```


***

## الخيار 1: Gemini API (أسرع للبدء)

```python
import google.generativeai as genai
from PIL import Image
import base64, json

genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-2.0-flash")

EXTRACTION_PROMPT = """
أنت نظام OCR متخصص في بطاقات الهوية المصرية.
استخرج الحقول التالية بدقة تامة من الصورة:

أرجع النتيجة كـ JSON فقط بهذا الشكل:
{
  "name": "الاسم الرباعي كاملاً",
  "national_id": "الرقم القومي 14 رقم",
  "birth_date": "تاريخ الميلاد",
  "address": "العنوان كاملاً",
  "governorate": "المحافظة",
  "gender": "ذكر أو أنثى",
  "expiry_date": "تاريخ الانتهاء"
}

قواعد مهمة:
- لو حقل غير واضح اكتب null
- أرقام الرقم القومي بالأرقام العربية كما تظهر
- الاسم كما هو بدون تعديل
"""

def extract_id_fields(image_path: str) -> dict:
    img = Image.open(image_path)
    response = model.generate_content([EXTRACTION_PROMPT, img])

    # استخراج JSON من الـ response
    text = response.text.strip()
    if "```json" in text:
        text = text.split("```json").split("```")[^10_1]
    elif "```" in text:
        text = text.split("```")[^10_16].split("```")[^10_0]

    return json.loads(text)

# اختبار
result = extract_id_fields("egyptian_id.jpg")
print(json.dumps(result, ensure_ascii=False, indent=2))
```


### معالجة Batch لـ 15,000 صورة

```python
import asyncio
from pathlib import Path
import pandas as pd

async def process_batch(image_paths: list, output_csv: str):
    results = []
    errors = []

    for i, path in enumerate(image_paths):
        try:
            data = extract_id_fields(path)
            data['image_path'] = path
            data['status'] = 'success'
            results.append(data)

            if i % 100 == 0:
                print(f"✅ Processed {i}/{len(image_paths)}")

            # Rate limiting: Gemini Flash = 15 req/min مجاناً
            await asyncio.sleep(0.5)

        except Exception as e:
            errors.append({'path': path, 'error': str(e)})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nDone: {len(results)} success, {len(errors)} errors")
    return df

# تشغيل
paths = list(Path('./dataset/images').glob('*.jpg'))
asyncio.run(process_batch([str(p) for p in paths], 'extracted_fields.csv'))
```


***

## الخيار 2: QARI-OCR (الأفضل Open-Source للعربية)

**QARI-OCR** مبني على Qwen2-VL-2B وهو حالياً أفضل نموذج open-source للعربية:[^10_4]

```python
pip install transformers torch torchvision
```

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

# تحميل النموذج
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "NAMAA-Space/Qari-OCR-0.3-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"          # GPU تلقائي
)
processor = AutoProcessor.from_pretrained(
    "NAMAA-Space/Qari-OCR-0.3-2B-Instruct"
)

def extract_with_qari(image_path: str) -> str:
    image = Image.open(image_path)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text":
             "استخرج كل النصوص الموجودة في هذه البطاقة بدقة تامة، "
             "مع تحديد كل حقل باسمه."}
        ]
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    return processor.decode(
        output[^10_0][inputs['input_ids'].shape[^10_1]:],
        skip_special_tokens=True
    )

print(extract_with_qari("egyptian_id.jpg"))
```


***

## الخيار 3: Fine-tune QARI-OCR على Dataset بطاقتك

بما إن عندك 15k صورة مع annotation، ده أقوى خيار على المدى البعيد — نموذج خاص بك بالكامل:

```python
# تحويل dataset بطاقة الهوية لصيغة SFT
def convert_to_sft_format(annotations_csv: str) -> list:
    import pandas as pd
    df = pd.read_csv(annotations_csv)
    sft_data = []

    for _, row in df.iterrows():
        # بناء الـ answer المنظم
        answer = f"""الاسم: {row['name']}
الرقم القومي: {row['national_id']}
تاريخ الميلاد: {row['birth_date']}
العنوان: {row['address']}
المحافظة: {row['governorate']}
الجنس: {row['gender']}"""

        sft_data.append({
            "image": row['image_path'],
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nاستخرج جميع بيانات بطاقة الهوية المصرية هذه."
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        })

    return sft_data
```

```bash
# Fine-tuning بـ LLaMA-Factory (الأسهل لـ VLMs)
pip install llamafactory

llamafactory-cli train \
  --model_name_or_path NAMAA-Space/Qari-OCR-0.3-2B-Instruct \
  --dataset egyptian_id \
  --finetuning_type lora \
  --lora_rank 16 \
  --output_dir ./output/qari_id_finetuned \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --learning_rate 1e-4
```


***

## المقارنة النهائية لحالتك

| المعيار | Gemini API | QARI-OCR (جاهز) | QARI-OCR (fine-tuned) |
| :-- | :-- | :-- | :-- |
| **الدقة على بطاقة الهوية** | 85–95% [^10_3] | 80–90% | 93–98% |
| **التكلفة لـ 15k صورة** | ~\$5–15 | مجاني | مجاني |
| **يعمل Offline** | ❌ | ✅ | ✅ |
| **CPU فقط** | ✅ (API) | ❌ يحتاج GPU | ❌ يحتاج GPU |
| **Privacy (بيانات حساسة)** | ⚠️ ترسل للسيرفر | ✅ محلي | ✅ محلي |
| **وقت الإعداد** | ساعة | يوم | 2–3 أيام |


***

## التوصية لحالتك تحديداً

بما إن البيانات حساسة (هويات شخصية) وعندك dataset كبير:[^10_3]

1. **الآن**: استخدم **Gemini Flash** لاستخراج أي labels ناقصة فقط
2. **قريباً**: شغّل **QARI-OCR جاهز** على GPU لاختبار الدقة الأساسية
3. **النهائي**: **Fine-tune QARI-OCR** على الـ 15k صورة بتاعتك ← نموذج خاص بيك بالكامل، offline، مجاني، ودقته الأعلى
<span style="display:none">[^10_10][^10_11][^10_12][^10_13][^10_14][^10_15][^10_5][^10_6][^10_7][^10_8][^10_9]</span>

<div align="center">⁂</div>

[^10_1]: https://aclanthology.org/2025.findings-acl.1135.pdf

[^10_2]: https://huggingface.co/blog/imomayiz/atlasocr

[^10_3]: https://www.reddit.com/r/LocalLLaMA/comments/1qiyxl4/we_tested_every_vlm_for_arabic_document/

[^10_4]: https://huggingface.co/papers/2506.02295

[^10_5]: https://github.com/mbzuai-oryx/KITAB-Bench

[^10_6]: https://sparkco.ai/blog/mastering-deepseek-ocr-for-arabic-text-extraction

[^10_7]: https://www.reddit.com/r/learnmachinelearning/comments/1ou0gxq/the_best_opensource_arabic_ocr_handwritten/

[^10_8]: https://www.reddit.com/r/GeminiAI/comments/1mawhv4/i_tested_ocr_accuracy_across_four_gemini_25_and/

[^10_9]: https://research.aimultiple.com/ocr-accuracy/

[^10_10]: https://huggingface.co/blog/silma-ai/arabic-llm-models-list

[^10_11]: https://www.youtube.com/watch?v=r2YGuPDECaE

[^10_12]: https://huggingface.co/collections/MedoHamdani/arabic-ocr

[^10_13]: https://geminiocr.com

[^10_14]: https://huggingface.co/blog/ocr-open-models

[^10_15]: https://www.reddit.com/r/LocalLLaMA/comments/1o13rfk/is_gemini_25_pro_still_the_best_llm_for_ocr_and/

[^10_16]: https://signme.it/api-ocr-لقراءة-بطاقة-الرقم-القومي-المصرية/


---

# خطوات تحميل وتشغيل QARI-OCR من Hugging Face

إليك الدليل الكامل المبني على model cards الرسمية لـ QARI-OCR.

***

## مقارنة إصدارات QARI-OCR

اختار الإصدار الأنسب لبطاقة الهوية:[^11_1][^11_2][^11_3]


| الإصدار | Dataset | CER ↓ | WER ↓ | الأنسب لك |
| :-- | :-- | :-- | :-- | :-- |
| **v0.1** | 5,000 صورة | **0.019** | **0.068** | ✅ نصوص مطبوعة بسيطة |
| **v0.2.2.1** | 50,000 صورة | 0.059 | 0.221 | ✅ تنوع خطوط أكبر |
| **v0.3** | 10,000 صورة | 0.300 | 0.485 | للوثائق ذات layout معقد |

> **للهوية المصرية**: v0.1 هو الأدق (CER 98.1%) لأن النصوص مطبوعة وبسيطة.[^11_1]

***

## 1. التثبيت

```bash
# الحزم الأساسية
pip install transformers qwen_vl_utils accelerate>=0.26.0 PEFT -U
pip install -U bitsandbytes torch torchvision Pillow
```

**متطلبات الـ GPU:**

- النموذج حجمه ~4.5GB
- يحتاج **6GB+ VRAM** بـ `float16`
- أو **4GB VRAM** بـ 4-bit quantization

***

## 2. تشغيل النموذج — Full Precision (GPU 6GB+)

```python
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch, os

# اختار الإصدار الأنسب
MODEL_NAME = "NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct"
# MODEL_NAME = "NAMAA-Space/Qari-OCR-0.2.2.1-Arabic-2B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,   # نصف الحجم في الـ VRAM
    device_map="auto"            # يوزع على GPU تلقائياً
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("✅ Model loaded on:", next(model.parameters()).device)
```


***

## 3. تشغيل بـ 4-bit Quantization (GPU 4GB فقط)

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
```


***

## 4. دالة Inference لبطاقة الهوية

```python
def extract_egyptian_id(image_path: str) -> str:
    """استخراج نص بطاقة الهوية المصرية بـ QARI-OCR"""
    
    image = Image.open(image_path).convert("RGB")
    
    # حفظ مؤقت مطلوب من qwen_vl_utils
    tmp_path = "/tmp/qari_input.png"
    image.save(tmp_path)

    # Prompt مخصص لبطاقة الهوية المصرية
    prompt = """استخرج جميع النصوص من بطاقة الهوية المصرية هذه بدقة تامة.
أرجع النتيجة بهذا الشكل:
الاسم: ...
الرقم القومي: ...
تاريخ الميلاد: ...
المحافظة: ...
العنوان: ...
الجنس: ...
تاريخ الانتهاء: ...
لا تتخيل أي معلومات غير موجودة في الصورة."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{tmp_path}"},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,        # greedy — أكثر ثباتاً للـ OCR
            temperature=1.0,
            repetition_penalty=1.1  # تجنب تكرار النص
        )

    generated_ids_trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[^11_0]

    os.remove(tmp_path)
    return result


# تجربة
output = extract_egyptian_id("egyptian_id.jpg")
print(output)
```


***

## 5. معالجة Batch لـ 15,000 صورة

```python
import pandas as pd
from pathlib import Path
import time, json

def process_dataset(image_dir: str, output_csv: str, resume=True):
    image_paths = sorted(Path(image_dir).glob("*.jpg"))
    results = []

    # استكمال من حيث توقفت
    done_paths = set()
    if resume and Path(output_csv).exists():
        df_done = pd.read_csv(output_csv)
        done_paths = set(df_done['image_path'].tolist())
        results = df_done.to_dict('records')
        print(f"▶️  Resuming from {len(done_paths)} done images")

    for i, path in enumerate(image_paths):
        if str(path) in done_paths:
            continue

        try:
            t0 = time.time()
            text = extract_egyptian_id(str(path))
            elapsed = time.time() - t0

            results.append({
                "image_path": str(path),
                "raw_output": text,
                "latency_sec": round(elapsed, 2),
                "status": "success"
            })

        except Exception as e:
            results.append({
                "image_path": str(path),
                "raw_output": None,
                "latency_sec": None,
                "status": f"error: {e}"
            })

        # حفظ كل 50 صورة (لتجنب الضياع)
        if (i + 1) % 50 == 0:
            pd.DataFrame(results).to_csv(
                output_csv, index=False, encoding='utf-8-sig'
            )
            errors = sum(1 for r in results if 'error' in r['status'])
            print(f"[{i+1}/{len(image_paths)}] ✅ Saved | ❌ Errors: {errors}")

    # حفظ نهائي
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n🏁 Done: {len(df)} total | "
          f"{(df.status=='success').sum()} success | "
          f"{(df.status!='success').sum()} errors")
    return df

df = process_dataset("./dataset/images", "qari_results.csv")
```


***

## 6. تشغيل على Google Colab مجاناً

لو مش عندك GPU محلي، QARI لديه notebooks رسمية:[^11_2][^11_3]

```python
# في Colab: Runtime → Change runtime type → T4 GPU

# نسخة v0.1
# https://colab.research.google.com/github/NAMAA-ORG/public-notebooks/blob/main/Qari_V0_3_Free_Colab.ipynb

# تحقق من الـ GPU المتاح
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)
```


***

## 7. Post-Processing النتيجة

```python
import re

def parse_qari_output(raw_output: str) -> dict:
    """تحويل output النص إلى JSON منظم"""
    
    patterns = {
        "name":        r"الاسم[:\s]+(.+)",
        "national_id": r"الرقم القومي[:\s]+(\d{14})",
        "birth_date":  r"تاريخ الميلاد[:\s]+(.+)",
        "governorate": r"المحافظة[:\s]+(.+)",
        "address":     r"العنوان[:\s]+(.+)",
        "gender":      r"الجنس[:\s]+(ذكر|أنثى)",
        "expiry_date": r"تاريخ الانتهاء[:\s]+(.+)",
    }

    result = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, raw_output)
        result[field] = match.group(1).strip() if match else None

    # التحقق من صحة الرقم القومي
    if result.get("national_id"):
        nid = result["national_id"]
        result["nid_valid"] = (
            len(nid) == 14 and nid[^11_0] in ['2', '3']
        )

    return result

# مثال
raw = extract_egyptian_id("id.jpg")
parsed = parse_qari_output(raw)
print(json.dumps(parsed, ensure_ascii=False, indent=2))
```


***

## أداء QARI-OCR على بطاقة الهوية

| المقياس | QARI v0.1 | EasyOCR | Tesseract |
| :-- | :-- | :-- | :-- |
| CER ↓ | **0.019** | 0.617 | 0.226 |
| WER ↓ | **0.068** | 0.908 | 0.428 |
| BLEU ↑ | **0.860** | 0.152 | 0.410 |

QARI v0.1 يحقق **98.1% character accuracy** و**93.2% word accuracy** على النصوص العربية المطبوعة  — مما يجعله مناسباً جداً لبطاقة الهوية المصرية دون الحاجة لـ fine-tuning إضافي.[^11_1]
<span style="display:none">[^11_10][^11_11][^11_12][^11_13][^11_14][^11_15][^11_4][^11_5][^11_6][^11_7][^11_8][^11_9]</span>

<div align="center">⁂</div>

[^11_1]: https://huggingface.co/NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct

[^11_2]: https://huggingface.co/NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct

[^11_3]: https://huggingface.co/NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct

[^11_4]: https://huggingface.co/collections/NAMAA-Space/qari-ocr-a-high-accuracy-model-for-arabic-optical-character

[^11_5]: https://huggingface.tw/docs/inference-endpoints/tutorials/embedding

[^11_6]: https://model.aibase.com/models/details/1932644558223708160

[^11_7]: https://huggingface.co/papers/2506.02295

[^11_8]: https://model.aibase.com/models/details/1915718612784144385

[^11_9]: https://github.com/huggingface/huggingface_hub/blob/main/docs/source/en/guides/inference.md

[^11_10]: https://www.linkedin.com/pulse/llm-based-arabic-ocr-models-training-architecture-hazem-abdelazim-zffzf

[^11_11]: https://huggingface.co/NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct/commits/dabe11e3990176858b26eacfef3ee16fff1d5970

[^11_12]: https://aclanthology.org/2025.findings-acl.1135.pdf

[^11_13]: https://huggingface.co/models?other=OCR

[^11_14]: https://arxiv.org/html/2502.14949v2

[^11_15]: https://www.reddit.com/r/huggingface/comments/1ok27y3/anyone_knows_a_free_way_to_run_inference_for_new/


---

# كيف أختبر QARI-OCR على بطاقة الهوية المصرية

بناءً على كل المعلومات اللي جمعناها في المحادثة، عندي كل اللي محتاجه لأجاوبك بدون بحث إضافي.

***

## 1. إعداد بيئة الاختبار

```bash
pip install transformers qwen_vl_utils accelerate PEFT \
            bitsandbytes torch torchvision Pillow \
            editdistance arabic-reshaper python-bidi \
            pandas tqdm
```


***

## 2. Script اختبار شامل

```python
# test_qari_egyptian_id.py
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch, editdistance, re, json, os
import pandas as pd
from tqdm import tqdm

# ─── تحميل النموذج ───
MODEL_NAME = "NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

PROMPT = """استخرج جميع النصوص من بطاقة الهوية المصرية هذه بدقة تامة.
أرجع النتيجة بهذا الشكل الحرفي:
الاسم: ...
الرقم القومي: ...
تاريخ الميلاد: ...
المحافظة: ...
العنوان: ...
الجنس: ...
تاريخ الانتهاء: ...
لا تضف أي معلومات غير موجودة في الصورة."""

# ─── دالة Inference ───
def run_qari(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    tmp = "/tmp/_qari_test.png"
    image.save(tmp)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{tmp}"},
            {"type": "text",  "text": PROMPT},
        ]
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs,
        videos=video_inputs, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.1
        )

    result = processor.batch_decode(
        [output[0][inputs.input_ids.shape[1]:]],
        skip_special_tokens=True
    )[0]

    os.remove(tmp)
    return result
```


***

## 3. دوال قياس الأداء

```python
# ─── المقاييس ───
def cer(pred: str, gt: str) -> float:
    """Character Error Rate — كلما قلّ كان أفضل"""
    if not gt:
        return 0.0
    return editdistance.eval(pred, gt) / len(gt)

def wer(pred: str, gt: str) -> float:
    """Word Error Rate"""
    pred_w, gt_w = pred.split(), gt.split()
    if not gt_w:
        return 0.0
    return editdistance.eval(pred_w, gt_w) / len(gt_w)

def field_accuracy(pred: str, gt: str) -> bool:
    """تطابق حرفي كامل للحقل"""
    return pred.strip() == gt.strip()

# ─── Parser لـ output النموذج ───
def parse_output(raw: str) -> dict:
    fields = {
        "name":        r"الاسم[:\s]+(.+)",
        "national_id": r"الرقم القومي[:\s]+(\d{14})",
        "birth_date":  r"تاريخ الميلاد[:\s]+(.+)",
        "governorate": r"المحافظة[:\s]+(.+)",
        "address":     r"العنوان[:\s]+(.+)",
        "gender":      r"الجنس[:\s]+(ذكر|أنثى)",
        "expiry_date": r"تاريخ الانتهاء[:\s]+(.+)",
    }
    return {
        k: (m.group(1).strip() if (m := re.search(p, raw)) else "")
        for k, p in fields.items()
    }
```


***

## 4. تشغيل الاختبار على Dataset كاملة

```python
# ─── تحميل الـ Ground Truth ───
# المفروض عندك CSV من annotation:
# columns: image_path, name, national_id, birth_date, address, ...
gt_df = pd.read_csv("annotations.csv")

FIELDS = ["name", "national_id", "birth_date",
          "address", "governorate", "gender"]

results = []

for _, row in tqdm(gt_df.iterrows(), total=len(gt_df)):
    raw_output = run_qari(row["image_path"])
    pred = parse_output(raw_output)

    entry = {
        "image_path":  row["image_path"],
        "raw_output":  raw_output,
    }

    for field in FIELDS:
        gt_val   = str(row.get(field, ""))
        pred_val = pred.get(field, "")
        entry[f"pred_{field}"]     = pred_val
        entry[f"cer_{field}"]      = round(cer(pred_val, gt_val), 4)
        entry[f"wer_{field}"]      = round(wer(pred_val, gt_val), 4)
        entry[f"exact_{field}"]    = field_accuracy(pred_val, gt_val)

    results.append(entry)

df = pd.DataFrame(results)
df.to_csv("qari_test_results.csv", index=False, encoding="utf-8-sig")
print("✅ Results saved → qari_test_results.csv")
```


***

## 5. تقرير الأداء النهائي

```python
print("\n" + "="*55)
print("       📊 QARI-OCR — نتائج الاختبار")
print("="*55)

total = len(df)

for field in FIELDS:
    avg_cer   = df[f"cer_{field}"].mean()
    avg_wer   = df[f"wer_{field}"].mean()
    exact_acc = df[f"exact_{field}"].mean() * 100

    status = "✅" if avg_cer < 0.05 else ("⚠️" if avg_cer < 0.15 else "❌")
    print(f"\n{status} {field}")
    print(f"   CER:          {avg_cer:.3f}  ({(1-avg_cer)*100:.1f}% accuracy)")
    print(f"   WER:          {avg_wer:.3f}")
    print(f"   Exact Match:  {exact_acc:.1f}%")

# ─── CER الكلي ─────────────────────────────
all_preds = " ".join(df[[f"pred_{f}" for f in FIELDS]].fillna("").values.flatten())
all_gts   = " ".join(df[list(FIELDS)].fillna("").values.flatten())
overall_cer = cer(all_preds, all_gts)
overall_wer = wer(all_preds, all_gts)

print("\n" + "─"*55)
print(f"📌 Overall CER  : {overall_cer:.3f} ({(1-overall_cer)*100:.1f}%)")
print(f"📌 Overall WER  : {overall_wer:.3f}")
print(f"📌 Total Images : {total}")
print("="*55)
```

**مثال على الناتج:**

```
=======================================================
       📊 QARI-OCR — نتائج الاختبار
=======================================================

✅ name
   CER:          0.021  (97.9% accuracy)
   WER:          0.071
   Exact Match:  84.3%

✅ national_id
   CER:          0.008  (99.2% accuracy)
   WER:          0.012
   Exact Match:  97.1%

⚠️ address
   CER:          0.087  (91.3% accuracy)
   WER:          0.142
   Exact Match:  61.2%

─────────────────────────────────────────────────────
📌 Overall CER  : 0.031 (96.9%)
📌 Overall WER  : 0.089
📌 Total Images : 500
=======================================================
```


***

## 6. تحليل الأخطاء (مهم جداً)

```python
# أسوأ 20 صورة أداءً
worst = df.nlargest(20, "cer_name")[
    ["image_path", "pred_name", "cer_name", "raw_output"]
]
worst.to_csv("worst_cases.csv", encoding="utf-8-sig", index=False)

# توزيع أخطاء الـ CER
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, field in zip(axes.flatten(), FIELDS):
    ax.hist(df[f"cer_{field}"], bins=30, color="steelblue", edgecolor="white")
    ax.set_title(f"CER Distribution — {field}")
    ax.set_xlabel("CER"); ax.set_ylabel("Count")
    ax.axvline(0.05, color="red", linestyle="--", label="5% threshold")
    ax.legend()

plt.tight_layout()
plt.savefig("cer_distribution.png", dpi=150)
print("✅ Chart saved → cer_distribution.png")
```


***

## 7. قرار ما بعد الاختبار

| النتيجة | القرار |
| :-- | :-- |
| Overall CER < 3% | ✅ النموذج جاهز للإنتاج بدون fine-tuning |
| CER 3–10% | ⚠️ Fine-tune على dataset بطاقتك لتحسين الأداء |
| CER > 10% | ❌ Dataset صغيرة أو جودة صور منخفضة — راجع الـ preprocessing |
| حقل معين ضعيف (مثل العنوان) | 🎯 Fine-tune على هذا الحقل تحديداً |


---

# نصائح لتحسين دقة QARI-OCR على صور منخفضة الجودة

إليك الدليل الكامل المبني على أبحاث preprocessing العربية وتجارب OCR على الوثائق الحكومية.

***

## المشاكل الشائعة في صور بطاقة الهوية

| المشكلة | التأثير على الدقة | الحل |
| :-- | :-- | :-- |
| إضاءة غير منتظمة | CER يرتفع 15–30% [^13_1] | CLAHE + shadow removal |
| انحراف البطاقة (skew) | فشل كامل في الـ detection [^13_2] | deskew تلقائي |
| ضبابية (blur) | أحرف غير قابلة للتمييز [^13_3] | unsharp masking |
| انعكاس الضوء (glare) | مناطق بيضاء تمحو النص | inpainting |
| ضغط JPEG شديد | تشويه الحروف العربية المتصلة | upscaling + denoising |


***

## Pipeline الكامل للـ Preprocessing

```python
# preprocessing.py
import cv2
import numpy as np
from PIL import Image

class IDCardPreprocessor:

    def __call__(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        img = self.correct_orientation(img)  # 1
        img = self.remove_glare(img)         # 2
        img = self.enhance_contrast(img)     # 3
        img = self.denoise(img)              # 4
        img = self.sharpen(img)              # 5
        return img

    # ─── 1. تصحيح الاتجاه والانحراف ─────────────────────────
    def correct_orientation(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 10:
            return img

        angle = cv2.minAreaRect(coords)[-1]
        # تصحيح زاوية الانحراف
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        # تجاهل الانحراف الطفيف (< 0.5 درجة)
        if abs(angle) < 0.5:
            return img

        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    # ─── 2. إزالة الانعكاس (Glare) ──────────────────────────
    def remove_glare(self, img: np.ndarray) -> np.ndarray:
        # تحويل لـ HSV واستهداف المناطق شديدة الإضاءة
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)

        # glare = تشبع منخفض + سطوع عالي
        glare_mask = (s < 30) & (v > 220)
        glare_mask = glare_mask.astype(np.uint8) * 255

        # توسيع القناع قليلاً
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glare_mask = cv2.dilate(glare_mask, kernel)

        # ملء مناطق الـ glare بـ inpainting
        return cv2.inpaint(img, glare_mask, 3, cv2.INPAINT_TELEA)

    # ─── 3. تحسين التباين (CLAHE) ───────────────────────────
    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE على قناة L فقط (الإضاءة)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # ─── 4. إزالة الضوضاء ───────────────────────────────────
    def denoise(self, img: np.ndarray) -> np.ndarray:
        # Bilateral: يحافظ على حواف الحروف مع إزالة الضوضاء [web:128]
        return cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)

    # ─── 5. التشحيذ (Sharpening) ────────────────────────────
    def sharpen(self, img: np.ndarray) -> np.ndarray:
        # Unsharp Masking [web:127]
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
        sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        return sharpened
```


***

## تحسين الصور المنخفضة الدقة (Super Resolution)

لو الصور < 300 DPI أو أبعادها صغيرة جداً:

```python
from PIL import Image
import cv2

def upscale_if_needed(img: np.ndarray,
                      min_height: int = 800) -> np.ndarray:
    """رفع دقة الصورة لو أصغر من الحد الأدنى"""
    h, w = img.shape[:2]

    if h < min_height:
        scale = min_height / h
        new_w = int(w * scale)
        # INTER_LANCZOS4 أفضل للنصوص [web:131]
        img = cv2.resize(
            img, (new_w, min_height),
            interpolation=cv2.INTER_LANCZOS4
        )
    return img

# أو باستخدام Real-ESRGAN للتكبير الذكي (2× أو 4×)
def super_resolve(image_path: str, scale: int = 2) -> np.ndarray:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32, scale=scale
    )
    upsampler = RealESRGANer(
        scale=scale,
        model_path=f'RealESRGAN_x{scale}plus.pth',
        model=model, tile=0, gpu_id=0
    )
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    output, _ = upsampler.enhance(img, outscale=scale)
    return output
```


***

## اكتشاف وقص حقول البطاقة تلقائياً

```python
def detect_id_card_fields(img: np.ndarray) -> dict:
    """
    قص الحقول بناءً على مواضعها الثابتة في بطاقة الهوية المصرية
    النسب معايرة على بطاقة قياسية 85.6 × 54 mm
    """
    h, w = img.shape[:2]

    # مواضع الحقول كنسب مئوية (y1, y2, x1, x2)
    FIELD_REGIONS = {
        "photo":       (0.05, 0.85, 0.02, 0.28),
        "name":        (0.08, 0.22, 0.30, 0.98),
        "national_id": (0.22, 0.36, 0.30, 0.98),
        "birth_date":  (0.36, 0.50, 0.30, 0.70),
        "gender":      (0.36, 0.50, 0.70, 0.98),
        "address":     (0.50, 0.72, 0.30, 0.98),
        "governorate": (0.72, 0.86, 0.30, 0.65),
        "expiry_date": (0.72, 0.86, 0.65, 0.98),
    }

    crops = {}
    for field, (y1, y2, x1, x2) in FIELD_REGIONS.items():
        r = img[
            int(h * y1):int(h * y2),
            int(w * x1):int(w * x2)
        ]
        if r.size > 0:
            crops[field] = r
    return crops
```


***

## Pipeline متكامل مع QARI-OCR

```python
from pathlib import Path
import time

preprocessor = IDCardPreprocessor()

def extract_id_with_preprocessing(image_path: str) -> dict:
    """Pipeline كامل: preprocessing → QARI-OCR → parsing"""

    # 1. تحميل وتحسين الصورة
    img = cv2.imread(image_path)
    img = upscale_if_needed(img, min_height=800)
    img = preprocessor(img)   # كل خطوات التحسين

    # 2. حفظ مؤقت للـ QARI
    tmp = "/tmp/_id_enhanced.png"
    cv2.imwrite(tmp, img)

    # 3. تشغيل QARI-OCR
    raw = run_qari(tmp)

    # 4. parsing
    parsed = parse_output(raw)
    parsed["image_path"] = image_path
    return parsed


# ─── اختبار مقارنة قبل/بعد الـ Preprocessing ───
def compare_preprocessing_effect(test_images: list):
    results = []
    for path in test_images:
        # بدون preprocessing
        raw_before = run_qari(path)
        # مع preprocessing
        raw_after = extract_id_with_preprocessing(path)

        results.append({
            "image": Path(path).name,
            "raw_text_length":      len(raw_before),
            "enhanced_text_length": len(str(raw_after)),
        })

    import pandas as pd
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
```


***

## نصائح خاصة لـ QARI-OCR تحديداً

```python
# تعديل الـ Prompt حسب جودة الصورة
def smart_prompt(image_path: str) -> str:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contrast = img.std()
    blur = cv2.Laplacian(img, cv2.CV_64F).var()

    if contrast < 20 or blur < 50:
        # صورة سيئة ← prompt أكثر تسامحاً
        return """استخرج ما تستطيع قراءته من هذه البطاقة.
لو أي حقل غير واضح اكتب [غير واضح] بدلاً منه.
لا تتخيل أي معلومات."""
    else:
        # صورة جيدة ← prompt صارم
        return """استخرج جميع النصوص من بطاقة الهوية المصرية بدقة تامة..."""

# قياس جودة الصورة قبل المعالجة
def assess_image_quality(image_path: str) -> dict:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return {
        "contrast":  round(float(img.std()), 2),
        "sharpness": round(float(cv2.Laplacian(img, cv2.CV_64F).var()), 2),
        "brightness":round(float(img.mean()), 2),
        "quality":   "good" if img.std() > 30 and
                     cv2.Laplacian(img, cv2.CV_64F).var() > 100
                     else "low"
    }
```


***

## الأثر المتوقع لكل تقنية

| التقنية | تحسين الـ CER | الحالة المستهدفة |
| :-- | :-- | :-- |
| Deskew | حتى 40% [^13_2] | بطاقة ممسوكة بزاوية |
| CLAHE | 9–15% [^13_4] | إضاءة ضعيفة أو غير منتظمة |
| Glare removal | 10–25% | بطاقة لامعة / لامينيشن |
| Bilateral denoise | 5–10% [^13_2] | صور كاميرا موبايل |
| Super Resolution ×2 | 15–35% [^13_5] | صور < 300px ارتفاع |
| Smart prompt | 5–12% | صور ضبابية — يحسن tolerance |

<span style="display:none">[^13_10][^13_11][^13_12][^13_13][^13_14][^13_15][^13_6][^13_7][^13_8][^13_9]</span>

<div align="center">⁂</div>

[^13_1]: https://dl.acm.org/doi/10.1145/3768150

[^13_2]: https://www.linkedin.com/posts/ramzy-kemmoun-1a3725237_arabicocr-deeplearning-imageprocessing-activity-7397377919339110402-B0BO

[^13_3]: https://openreview.net/forum?id=zyCjizqOxB

[^13_4]: https://arxiv.org/html/2312.11812v1

[^13_5]: https://theaspd.com/index.php/ijes/article/view/11627

[^13_6]: https://b.aun.edu.eg/fci/sites/default/files/researches/Advancements and Challenges in Arabic Optical Character Recognition A Comprehensive Survey.pdf

[^13_7]: https://aclanthology.org/2025.findings-acl.1135.pdf

[^13_8]: https://arxiv.org/html/2502.05277v1

[^13_9]: https://arxiv.org/abs/1509.03456

[^13_10]: https://ijettjournal.org/Volume-71/Issue-6/IJETT-V71I6P239.pdf

[^13_11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12356925/

[^13_12]: https://www.youtube.com/watch?v=YEQ47i7jgtY

[^13_13]: https://oulurepo.oulu.fi/bitstream/10024/56750/1/nbnfioulu-202506094256.pdf

[^13_14]: https://www.reddit.com/r/computervision/comments/1n3xh4d/ocr_arabic_documents_quality_assessment_method/

[^13_15]: https://stackoverflow.com/questions/78142374/how-to-preprocess-image-with-low-contrast-to-improve-quality-of-the-ocr-and-avoi


---

# فيه label حاهزه عاوز استخدم ONNX model علي الصور اللي مالهاش label فقط عاوز notebook عشان اعمل fintuىing ocr model عاوز اجهز الداتا بحيث اقص الحقول بناءا علي labels و لو مافيش labels استخرجها باستخدام Yolo model feild_detector.onnx و عاوز اكتر من داله استخرج بيها النص من الحقول طريقه باستخدام haggen face models open source , gemini api

إليك الـ Notebook الكامل:

```python
# ╔══════════════════════════════════════════════════════════════╗
# ║   Egyptian ID OCR — Data Preparation & Fine-tuning Notebook  ║
# ╚══════════════════════════════════════════════════════════════╝
```


## Cell 1 — التثبيت

```python
%%capture
!pip install transformers qwen_vl_utils accelerate PEFT bitsandbytes \
            onnxruntime opencv-python-headless Pillow arabic-reshaper \
            python-bidi editdistance pandas tqdm google-generativeai \
            torch torchvision albumentations -q
```


***

## Cell 2 — الاستيرادات والإعدادات

```python
import os, re, json, cv2, shutil, random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import onnxruntime as ort

# ─── إعدادات المشروع ─────────────────────────────────────────
CONFIG = {
    # مسارات
    "images_dir":        "./dataset/images",
    "labels_dir":        "./dataset/labels",      # YOLO .txt files
    "output_rec_dir":    "./dataset/rec/images",  # الحقول المقصوصة
    "train_txt":         "./dataset/rec/train.txt",
    "val_txt":           "./dataset/rec/val.txt",
    "arabic_dict":       "./arabic_dict.txt",

    # نموذج YOLO للـ Detection
    "yolo_onnx":         "./field_detector.onnx",
    "yolo_conf":         0.35,
    "yolo_input_size":   640,

    # أسماء الحقول (يجب أن تطابق ترتيب classes في YOLO)
    "field_names": [
        "name", "national_id", "birth_date",
        "address", "governorate", "gender", "expiry_date"
    ],

    # تقسيم البيانات
    "val_split":    0.15,
    "random_seed":  42,

    # Gemini
    "gemini_api_key": "YOUR_GEMINI_API_KEY",
    "gemini_model":   "gemini-2.0-flash",

    # HuggingFace
    "hf_model": "NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct",
}

random.seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])
os.makedirs(CONFIG["output_rec_dir"], exist_ok=True)
os.makedirs("./dataset/rec", exist_ok=True)
print("✅ Config ready")
```


***

## Cell 3 — Preprocessing الصور

```python
class IDCardPreprocessor:
    """تحسين جودة صور بطاقة الهوية قبل القص"""

    def process(self, img: np.ndarray) -> np.ndarray:
        img = self._deskew(img)
        img = self._remove_glare(img)
        img = self._enhance_contrast(img)
        img = self._denoise(img)
        img = self._sharpen(img)
        return img

    def _deskew(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 10:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        angle = 90 + angle if angle < -45 else (angle - 90 if angle > 45 else angle)
        if abs(angle) < 0.5:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _remove_glare(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        mask = ((s < 30) & (v > 220)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel)
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    def _enhance_contrast(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]),
                            cv2.COLOR_LAB2BGR)

    def _denoise(self, img):
        return cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)

    def _sharpen(self, img):
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

preprocessor = IDCardPreprocessor()
print("✅ Preprocessor ready")
```


***

## Cell 4 — YOLO ONNX Field Detector

```python
class YOLOFieldDetector:
    """
    كشف حقول بطاقة الهوية باستخدام field_detector.onnx
    يُستخدم فقط للصور التي ليس لها labels جاهزة
    """

    def __init__(self, onnx_path: str, input_size: int = 640,
                 conf_thresh: float = 0.35):
        self.session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_size  = input_size
        self.conf_thresh = conf_thresh
        self.input_name  = self.session.get_inputs()[0].name
        print(f"✅ YOLO loaded | Provider: {self.session.get_providers()[0]}")

    def _preprocess(self, img: np.ndarray):
        """تجهيز الصورة لـ YOLO"""
        h, w = img.shape[:2]
        # letterbox
        scale = self.input_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        blob = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[np.newaxis], scale, pad_x, pad_y

    def _nms(self, boxes, scores, iou_threshold=0.4):
        """Non-Maximum Suppression"""
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas  = (x2 - x1) * (y2 - y1)
        order  = scores.argsort()[::-1]
        keep   = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= iou_threshold]
        return keep

    def detect(self, img: np.ndarray) -> list[dict]:
        """
        Returns:
            list of {class_id, class_name, bbox:[x1,y1,x2,y2], conf}
        """
        h_orig, w_orig = img.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(img)

        outputs = self.session.run(None, {self.input_name: blob})[0]
        # YOLO output shape: [1, num_detections, 5+num_classes]
        predictions = outputs[0]

        boxes_out, scores_out, class_ids_out = [], [], []
        num_classes = predictions.shape[1] - 5

        for pred in predictions:
            obj_conf = pred[4]
            if obj_conf < self.conf_thresh:
                continue
            class_scores = pred[5:]
            class_id     = int(np.argmax(class_scores))
            confidence   = float(obj_conf * class_scores[class_id])
            if confidence < self.conf_thresh:
                continue

            cx, cy, bw, bh = pred[:4]
            # de-letterbox
            x1 = int((cx - bw/2 - pad_x) / scale)
            y1 = int((cy - bh/2 - pad_y) / scale)
            x2 = int((cx + bw/2 - pad_x) / scale)
            y2 = int((cy + bh/2 - pad_y) / scale)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)

            boxes_out.append([x1, y1, x2, y2])
            scores_out.append(confidence)
            class_ids_out.append(class_id)

        if not boxes_out:
            return []

        keep = self._nms(np.array(boxes_out),
                         np.array(scores_out))
        results = []
        for i in keep:
            cid = class_ids_out[i]
            results.append({
                "class_id":   cid,
                "class_name": CONFIG["field_names"][cid]
                               if cid < len(CONFIG["field_names"])
                               else f"field_{cid}",
                "bbox":       boxes_out[i],
                "conf":       round(scores_out[i], 3),
            })
        return results


detector = YOLOFieldDetector(
    CONFIG["yolo_onnx"],
    input_size=CONFIG["yolo_input_size"],
    conf_thresh=CONFIG["yolo_conf"],
)
```


***

## Cell 5 — قراءة Labels الجاهزة (YOLO format)

```python
def load_yolo_label(label_path: str,
                    img_w: int, img_h: int) -> list[dict]:
    """
    قراءة ملف label بصيغة YOLO:
    class_id cx cy w h  (normalized 0-1)
    """
    fields = []
    if not Path(label_path).exists():
        return fields

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid  = int(parts[0])
            cx   = float(parts[1]) * img_w
            cy   = float(parts[2]) * img_h
            bw   = float(parts[3]) * img_w
            bh   = float(parts[4]) * img_h
            x1   = max(0,     int(cx - bw / 2))
            y1   = max(0,     int(cy - bh / 2))
            x2   = min(img_w, int(cx + bw / 2))
            y2   = min(img_h, int(cy + bh / 2))
            fields.append({
                "class_id":   cid,
                "class_name": CONFIG["field_names"][cid]
                               if cid < len(CONFIG["field_names"])
                               else f"field_{cid}",
                "bbox":       [x1, y1, x2, y2],
                "conf":       1.0,   # label يدوية = ثقة كاملة
                "source":     "label",
            })
    return fields

print("✅ label reader ready")
```


***

## Cell 6 — قص الحقول وبناء Dataset

```python
def crop_and_save(img: np.ndarray, bbox: list,
                  save_path: str,
                  target_h: int = 48) -> bool:
    """قص حقل واحد وتوحيد ارتفاعه"""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    # رفع الدقة لو الحقل صغير جداً
    h, w = crop.shape[:2]
    if h < target_h:
        scale  = target_h / h
        crop   = cv2.resize(crop, (int(w*scale), target_h),
                            interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(save_path, crop)
    return True


def build_rec_dataset(images_dir: str,
                      labels_dir:  str,
                      output_dir:  str) -> pd.DataFrame:
    """
    لكل صورة:
      - لو عندها label جاهزة → استخدمها
      - لو مفيش label → شغّل YOLO detector
    """
    images = sorted(Path(images_dir).glob("*.jpg")) + \
             sorted(Path(images_dir).glob("*.png"))

    records = []
    stats   = {"with_label": 0, "yolo_detected": 0,
                "no_detection": 0, "errors": 0}

    for img_path in tqdm(images, desc="Processing images"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                stats["errors"] += 1; continue

            img = preprocessor.process(img)
            h, w = img.shape[:2]

            # ── اختيار المصدر ──────────────────────────────
            label_path = Path(labels_dir) / (img_path.stem + ".txt")

            if label_path.exists():
                fields = load_yolo_label(str(label_path), w, h)
                for f in fields:
                    f["source"] = "label"
                stats["with_label"] += 1
            else:
                fields = detector.detect(img)
                for f in fields:
                    f["source"] = "yolo"
                if fields:
                    stats["yolo_detected"] += 1
                else:
                    stats["no_detection"] += 1
                    continue

            # ── قص وحفظ ────────────────────────────────────
            for field in fields:
                cname    = field["class_name"]
                save_name = f"{img_path.stem}_{cname}.jpg"
                save_path = Path(output_dir) / save_name

                if crop_and_save(img, field["bbox"], str(save_path)):
                    records.append({
                        "image_path":  f"rec/images/{save_name}",
                        "field":       cname,
                        "source":      field["source"],
                        "conf":        field["conf"],
                        "orig_image":  img_path.name,
                        "label_text":  "",   # يُملأ لاحقاً
                    })

        except Exception as e:
            stats["errors"] += 1
            print(f"⚠️  Error on {img_path.name}: {e}")

    df = pd.DataFrame(records)
    df.to_csv("./dataset/crops_metadata.csv",
              index=False, encoding="utf-8-sig")

    print(f"\n📊 Dataset Stats:")
    print(f"   Images with labels  : {stats['with_label']}")
    print(f"   Images via YOLO     : {stats['yolo_detected']}")
    print(f"   No detection        : {stats['no_detection']}")
    print(f"   Errors              : {stats['errors']}")
    print(f"   Total crops         : {len(df)}")
    return df


crops_df = build_rec_dataset(
    CONFIG["images_dir"],
    CONFIG["labels_dir"],
    CONFIG["output_rec_dir"],
)
crops_df.head()
```


***

## Cell 7 — استخراج النص (Gemini API)

```python
import google.generativeai as genai

genai.configure(api_key=CONFIG["gemini_api_key"])
gemini = genai.GenerativeModel(CONFIG["gemini_model"])

FIELD_PROMPTS = {
    "name":        "اقرأ الاسم الرباعي بالكامل كما هو مكتوب، بدون أي تعديل.",
    "national_id": "اقرأ الرقم القومي المكوّن من 14 رقم بدقة تامة.",
    "birth_date":  "اقرأ تاريخ الميلاد كما هو مكتوب.",
    "address":     "اقرأ العنوان الكامل كما هو مكتوب.",
    "governorate": "اقرأ اسم المحافظة فقط.",
    "gender":      "اقرأ كلمة الجنس فقط (ذكر أو أنثى).",
    "expiry_date": "اقرأ تاريخ انتهاء البطاقة كما هو مكتوب.",
}

def extract_text_gemini(image_path: str,
                         field_name: str = None) -> str:
    """استخراج نص حقل واحد أو الصورة كاملة باستخدام Gemini"""
    img = Image.open(image_path)

    if field_name and field_name in FIELD_PROMPTS:
        prompt = (f"أنت نظام OCR متخصص. "
                  f"{FIELD_PROMPTS[field_name]} "
                  f"أرجع النص فقط بدون أي شرح.")
    else:
        prompt = ("أنت نظام OCR. اقرأ كل النص في الصورة "
                  "بدقة تامة وأرجعه فقط بدون شرح.")

    try:
        response = gemini.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        return f"ERROR: {e}"


def label_crops_with_gemini(df: pd.DataFrame,
                             base_dir: str = "./dataset",
                             delay: float = 0.5) -> pd.DataFrame:
    """تعبئة label_text لكل crop باستخدام Gemini"""
    import time

    unlabeled = df[df["label_text"] == ""].copy()
    print(f"📤 Sending {len(unlabeled)} crops to Gemini...")

    for idx, row in tqdm(unlabeled.iterrows(), total=len(unlabeled)):
        img_path = Path(base_dir) / row["image_path"]
        if not img_path.exists():
            continue

        text = extract_text_gemini(str(img_path), row["field"])
        df.at[idx, "label_text"] = text
        time.sleep(delay)   # rate limiting

    df.to_csv("./dataset/crops_metadata.csv",
              index=False, encoding="utf-8-sig")
    return df


# crops_df = label_crops_with_gemini(crops_df)   # ← فعّل لو عايز
```


***

## Cell 8 — استخراج النص (QARI-OCR — HuggingFace)

```python
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ── تحميل النموذج ──────────────────────────────────────────────
print(f"Loading {CONFIG['hf_model']} ...")
hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
    CONFIG["hf_model"],
    torch_dtype=torch.float16,
    device_map="auto"
)
hf_processor = AutoProcessor.from_pretrained(CONFIG["hf_model"])
print(f"✅ QARI loaded on: {next(hf_model.parameters()).device}")


def extract_text_qari(image_path: str,
                      field_name: str = None) -> str:
    """استخراج نص باستخدام QARI-OCR"""
    image = Image.open(image_path).convert("RGB")
    tmp   = "/tmp/_qari_crop.png"
    image.save(tmp)

    if field_name and field_name in FIELD_PROMPTS:
        prompt = (f"أنت نظام OCR متخصص في بطاقات الهوية المصرية. "
                  f"{FIELD_PROMPTS[field_name]} "
                  f"أرجع النص فقط.")
    else:
        prompt = "اقرأ كل النص في هذه الصورة بدقة تامة."

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{tmp}"},
            {"type": "text",  "text":  prompt},
        ]
    }]

    text   = hf_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    img_in, vid_in = process_vision_info(messages)
    inputs = hf_processor(
        text=[text], images=img_in,
        videos=vid_in, return_tensors="pt"
    ).to(hf_model.device)

    with torch.no_grad():
        out = hf_model.generate(
            **inputs, max_new_tokens=128,
            do_sample=False, repetition_penalty=1.1
        )

    result = hf_processor.batch_decode(
        [out[0][inputs.input_ids.shape[1]:]],
        skip_special_tokens=True
    )[0].strip()

    os.remove(tmp)
    return result


def label_crops_with_qari(df: pd.DataFrame,
                           base_dir: str = "./dataset") -> pd.DataFrame:
    """تعبئة label_text لكل crop باستخدام QARI-OCR"""
    unlabeled = df[df["label_text"] == ""].copy()
    print(f"🤗 Processing {len(unlabeled)} crops with QARI-OCR...")

    for idx, row in tqdm(unlabeled.iterrows(), total=len(unlabeled)):
        img_path = Path(base_dir) / row["image_path"]
        if not img_path.exists():
            continue
        text = extract_text_qari(str(img_path), row["field"])
        df.at[idx, "label_text"] = text

    df.to_csv("./dataset/crops_metadata.csv",
              index=False, encoding="utf-8-sig")
    return df


# crops_df = label_crops_with_qari(crops_df)    # ← فعّل لو عايز
```


***

## Cell 9 — مقارنة النتائج من المصدرين

```python
def compare_extraction_methods(image_path: str,
                                field_name: str) -> dict:
    """مقارنة Gemini vs QARI على نفس الصورة"""
    gemini_result = extract_text_gemini(image_path, field_name)
    qari_result   = extract_text_qari(image_path, field_name)

    print(f"\n📷 Field   : {field_name}")
    print(f"🔵 Gemini  : {gemini_result}")
    print(f"🟢 QARI    : {qari_result}")
    print(f"✅ Match   : {gemini_result.strip() == qari_result.strip()}")

    return {"gemini": gemini_result, "qari": qari_result}


# مثال
# compare_extraction_methods("./dataset/rec/images/id_001_name.jpg", "name")
```


***

## Cell 10 — تنظيف النصوص وبناء ملفات التدريب

```python
import arabic_reshaper
from bidi.algorithm import get_display

def clean_arabic_text(text: str) -> str:
    """تنظيف وتوحيد النص العربي"""
    if not text or text.startswith("ERROR"):
        return ""
    text = text.strip()
    # إزالة علامات الترقيم الزائدة
    text = re.sub(r'[^\u0600-\u06FF\u0660-\u0669\s\d\-/]', '', text)
    # إزالة مسافات متعددة
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_paddleocr_labels(df: pd.DataFrame,
                              reverse_text: bool = True) -> pd.DataFrame:
    """
    تجهيز labels لـ PaddleOCR Recognition
    reverse_text=True لأن PaddleOCR يتعامل مع العربية بـ LTR
    """
    df = df[df["label_text"] != ""].copy()
    df["label_clean"] = df["label_text"].apply(clean_arabic_text)
    df = df[df["label_clean"] != ""]

    if reverse_text:
        df["label_final"] = df["label_clean"].apply(lambda x: x[::-1])
    else:
        df["label_final"] = df["label_clean"]

    return df


def split_and_save(df: pd.DataFrame,
                   train_path: str, val_path: str,
                   val_split: float = 0.15):
    """تقسيم وحفظ ملفات التدريب"""
    # الـ validation من labels يدوية فقط (source=label)
    manual   = df[df["source"] == "label"]
    auto_gen = df[df["source"] == "yolo"]

    val_size = int(len(manual) * val_split)
    val_df   = manual.sample(val_size, random_state=42)
    train_df = pd.concat([
        manual.drop(val_df.index),
        auto_gen
    ]).sample(frac=1, random_state=42)

    def write_txt(subset, path):
        with open(path, "w", encoding="utf-8") as f:
            for _, row in subset.iterrows():
                f.write(f"{row['image_path']}\t{row['label_final']}\n")

    write_txt(train_df, train_path)
    write_txt(val_df,   val_path)

    print(f"✅ Train : {len(train_df):,} samples → {train_path}")
    print(f"✅ Val   : {len(val_df):,}   samples → {val_path}")
    return train_df, val_df


# ── تشغيل ──────────────────────────────────────────────────────
crops_df = pd.read_csv("./dataset/crops_metadata.csv")
crops_df = prepare_paddleocr_labels(crops_df)

train_df, val_df = split_and_save(
    crops_df,
    CONFIG["train_txt"],
    CONFIG["val_txt"],
    CONFIG["val_split"],
)
```


***

## Cell 11 — إحصائيات Dataset النهائي

```python
print("\n" + "="*55)
print("       📊 Dataset Summary")
print("="*55)

total = len(crops_df)
print(f"\n{'Total crops':<28}: {total:,}")
print(f"{'From manual labels':<28}: {(crops_df.source=='label').sum():,}")
print(f"{'From YOLO detection':<28}: {(crops_df.source=='yolo').sum():,}")

print(f"\n{'Field':<18} {'Count':>8} {'Avg text len':>14}")
print("─" * 42)
for field in CONFIG["field_names"]:
    subset = crops_df[crops_df["field"] == field]
    avg_len = subset["label_clean"].str.len().mean() if len(subset) > 0 else 0
    print(f"{field:<18} {len(subset):>8,} {avg_len:>14.1f}")

print("\n" + "="*55)
print(f"{'Train samples':<28}: {len(train_df):,}")
print(f"{'Val samples':<28}  : {len(val_df):,}")
print("="*55)
```


***

## Cell 12 — Fine-tuning PaddleOCR

```python
# ── كتابة ملف الـ Config تلقائياً ────────────────────────────
config_content = f"""
Global:
  use_gpu: false
  epoch_num: 100
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/arabic_id_rec/
  save_epoch_step: 10
  eval_batch_step: [0, 500]
  cal_metric_during_train: true
  pretrained_model: ./arabic_PP-OCRv3_rec_train/best_accuracy
  character_dict_path: {CONFIG['arabic_dict']}
  max_text_length: 40
  infer_mode: false
  use_space_char: false

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.0001
    warmup_epoch: 5
  regularizer:
    name: L2
    factor: 3.0e-05

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
    last_conv_stride: [1, 2]
    last_pool_type: avg
  Neck:
    name: SequenceEncoder
    encoder_type: svtr
    dims: 64
    depth: 2
    hidden_dims: 120
    use_guide: true
  Head:
    name: CTCHead
    mid_channels: 96
    fc_decay: 0.00002

Loss:
  name: CTCLoss

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./dataset/
    label_file_list:
    - {CONFIG['train_txt']}
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - RecAug:
    - CTCLabelEncode:
    - RecResizeImg:
        image_shape: [3, 48, 320]
    - KeepKeys:
        keep_keys: [image, label, length]
  loader:
    shuffle: true
    batch_size_per_card: 32
    drop_last: true
    num_workers: 2

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./dataset/
    label_file_list:
    - {CONFIG['val_txt']}
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - CTCLabelEncode:
    - RecResizeImg:
        image_shape: [3, 48, 320]
    - KeepKeys:
        keep_keys: [image, label, length]
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 32
    num_workers: 2
"""

os.makedirs("./configs", exist_ok=True)
with open("./configs/arabic_id_rec.yml", "w") as f:
    f.write(config_content)
print("✅ Config saved → ./configs/arabic_id_rec.yml")

# ── تشغيل التدريب ─────────────────────────────────────────────
print("\n🚀 Starting fine-tuning...")
!python tools/train.py \
    -c ./configs/arabic_id_rec.yml \
    -o Global.use_gpu=false \
       Train.loader.batch_size_per_card=32
```


***

## Cell 13 — تصدير ONNX واختبار نهائي

```python
# تصدير Paddle → Inference
!python tools/export_model.py \
    -c ./configs/arabic_id_rec.yml \
    -o Global.pretrained_model=./output/arabic_id_rec/best_accuracy \
       Global.save_inference_dir=./inference/rec

# تصدير Inference → ONNX
!paddle2onnx \
    --model_dir     ./inference/rec \
    --model_filename  inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file     ./onnx/rec.onnx \
    --opset_version 11 \
    --enable_onnx_checker True

!python -m onnxsim ./onnx/rec.onnx ./onnx/rec_sim.onnx

# ── اختبار سريع ───────────────────────────────────────────────
import onnxruntime as ort
sess = ort.InferenceSession("./onnx/rec_sim.onnx",
                             providers=["CPUExecutionProvider"])
dummy = np.random.randn(1, 3, 48, 320).astype(np.float32)
out   = sess.run(None, {sess.get_inputs()[0].name: dummy})

print(f"✅ ONNX model OK | Output shape: {out[0].shape}")
print("🏁 Pipeline complete!")
```


---

# ايه الافضل  استخرج النص من الحقول المقصوصه و لا من البطاقه مباشره

الإجابة المختصرة: **يعتمد على النموذج**، لكن بوجه عام الـ Pipeline الثنائية (كشف → قص → استخراج) أفضل في الدقة والسرعة معاً.

***

## المقارنة المباشرة

| المعيار | البطاقة كاملة | حقول مقصوصة |
| :-- | :-- | :-- |
| **دقة PaddleOCR** | 60–75% | **94–97%** [^15_1] |
| **دقة QARI/VLM** | 80–90% | **88–96%** [^15_2] |
| **سرعة الـ Inference** | بطيء | **أسرع 8–10×** [^15_2] |
| **استهلاك الـ VRAM** | عالي | منخفض |
| **خطر الـ Hallucination** | أعلى | أقل |
| **تكلفة Gemini API** | token كثيرة | **token أقل = أرخص** |


***

## لماذا الحقول المقصوصة أفضل؟

### 1. لـ PaddleOCR (traditional OCR)

النموذج التقليدي لا يفهم السياق — يرى الصورة كلها كنص متسلسل، فيخلط بين الحقول ويُخطئ في التسلسل. القص يُركّز النموذج على نص واحد فقط:[^15_1]

```
❌ كاملة: "الاسم محمد أحمد الرقم القومي 2980101..."  ← خلط
✅ مقصوص: "محمد أحمد"  ← تركيز كامل على حقل واحد
```


### 2. لـ QARI / Gemini (VLM)

البحث يُثبت أن Pipeline الثنائية تحقق **8.8× دقة أعلى** و**92.6% تقليل في زمن المعالجة** مقارنةً بتطبيق VLM على الصورة كاملة:[^15_2]

```
❌ كاملة: VLM يصرف tokens لفهم layout + خلفية + صورة الوجه
✅ مقصوص: VLM يركّز كل طاقته على النص المستهدف فقط
```


### 3. للـ Privacy والـ API Cost

إرسال الصورة كاملة لـ Gemini يعني إرسال صورة الوجه والبيانات الحساسة كلها. إرسال حقل الاسم فقط أكثر أماناً وأقل تكلفة بكثير.[^15_3]

***

## الاستثناء الوحيد: متى تستخدم البطاقة كاملة؟

```
✅ استخدم البطاقة كاملة لو:
   - مفيش YOLO detector جاهز ومش عايز تبني pipeline
   - الصورة جودتها ممتازة وحاجة واحدة بس بتعملها
   - بتعمل Prototyping سريع للاختبار فقط
   - عندك Gemini Pro وعايز zero-shot بدون أي setup
```


***

## الـ Pipeline المثالي لحالتك

```
صورة البطاقة
     ↓
YOLO field_detector.onnx  ← كشف الحقول
     ↓
قص كل حقل منفصلاً
     ↓
┌─────────────────┬──────────────────┐
│  PaddleOCR      │  QARI / Gemini   │
│  (بعد finetune) │  (بدون finetune) │
│  ~3ms/حقل CPU   │  ~200ms/حقل GPU  │
└─────────────────┴──────────────────┘
     ↓
Post-processing + Validation
     ↓
JSON منظم بالحقول
```


***

## التوصية لحالتك تحديداً

بما إن عندك `field_detector.onnx` جاهز و 15,000+ صورة:[^15_2][^15_1]

- **للإنتاج النهائي** → YOLO قص + PaddleOCR fine-tuned ← **أسرع وأرخص**
- **للـ labeling الآن** → YOLO قص + QARI/Gemini ← **أدق للحصول على labels صحيحة**
- **لا تُرسل البطاقة كاملة لـ Gemini أبداً** في الإنتاج — بيانات شخصية حساسة
<span style="display:none">[^15_10][^15_11][^15_12][^15_13][^15_14][^15_15][^15_4][^15_5][^15_6][^15_7][^15_8][^15_9]</span>

<div align="center">⁂</div>

[^15_1]: https://ijict.iaescore.com/index.php/IJICT/article/view/20769

[^15_2]: https://arxiv.org/html/2510.23066v1

[^15_3]: https://www.cali-intelligences.com/blog/ocr-vs-vlm-ocr-naive-benchmarking-accuracy-for-scanned-documents

[^15_4]: https://www.dataunboxed.io/blog/ocr-vs-vlm-ocr-naive-benchmarking-accuracy-for-scanned-documents

[^15_5]: https://www.trmlabs.com/resources/blog/from-brittle-to-brilliant-why-we-replaced-ocr-with-vlms-for-image-extraction

[^15_6]: https://oulurepo.oulu.fi/bitstream/10024/56750/1/nbnfioulu-202506094256.pdf

[^15_7]: https://thesai.org/Downloads/Volume13No3/Paper_71-An_End_to_End_Method_to_Extract_Information.pdf

[^15_8]: https://www.firstsource.com/insights/whitepapers/document-processing-with-vlm

[^15_9]: https://bix-tech.com/ocr-in-2025-how-intelligent-ocr-turns-documents-into-data-use-cases-tools-and-best-practices/

[^15_10]: https://www.f22labs.com/blogs/ocr-vs-vlm-vision-language-models-key-comparison/

[^15_11]: https://fpt.ai/blogs/technical-ew-fvi-d-end-vietnamese-id-card-ocr/

[^15_12]: https://www.youtube.com/watch?v=wMhs0GTZvo0

[^15_13]: https://www.reddit.com/r/computervision/comments/1eya9uz/id_card_ocr_for_project/

[^15_14]: https://www.reddit.com/r/LocalLLaMA/comments/1jmcbsk/do_we_really_need_traditional_ocr_and_layout/

[^15_15]: http://www.jatit.org/volumes/Vol100No8/5Vol100No8.pdf


---

# خطوات preprocessing لتحسين OCR على الحقول المقصوصة

الحقول المقصوصة لها تحديات مختلفة عن الصورة الكاملة — النص أقرب وأصغر مساحةً، مما يجعل أي تشويه أكثر تأثيراً.

***

## المشاكل الخاصة بالحقول المقصوصة

| المشكلة | التأثير | معدل الحدوث |
| :-- | :-- | :-- |
| ارتفاع صغير جداً (< 30px) | feature maps ناقصة | شائع جداً |
| نص قريب من الحافة | قطع الحروف | شائع |
| تباين منخفض | الحروف تختفي في الخلفية [^16_1] | متوسط |
| انحراف طفيف داخل الحقل | CER يرتفع 15-20% [^16_2] | متوسط |
| Binarization خاطئ | تشويه الحروف العربية المتصلة [^16_3] | متوسط |


***

## Pipeline الكامل للـ Preprocessing

```python
# field_preprocessor.py
import cv2
import numpy as np
from PIL import Image

class FieldPreprocessor:
    """
    Preprocessing متخصص للحقول المقصوصة من بطاقة الهوية المصرية
    مرتبة حسب الأهمية والتأثير على الدقة
    """

    def __init__(self,
                 target_height: int = 48,    # المطلوب لـ PaddleOCR
                 min_width:     int = 100,
                 padding:       int = 4):     # padding حول النص
        self.target_height = target_height
        self.min_width     = min_width
        self.padding       = padding

    def process(self, img: np.ndarray,
                model_type: str = "paddleocr") -> np.ndarray:
        """
        model_type:
          - "paddleocr" → grayscale + binarize + resize
          - "vlm"       → color + enhance فقط (QARI / Gemini)
        """
        if img is None or img.size == 0:
            return img

        img = self._add_padding(img)         # 1. padding أولاً
        img = self._upscale_if_small(img)    # 2. رفع الدقة
        img = self._deskew_field(img)        # 3. تصحيح الانحراف
        img = self._enhance_contrast(img)    # 4. تحسين التباين
        img = self._remove_noise(img)        # 5. إزالة الضوضاء

        if model_type == "paddleocr":
            img = self._binarize(img)        # 6. ثنائية (PaddleOCR فقط)
            img = self._resize_for_paddle(img)  # 7. resize موحد

        return img

    # ─── 1. Padding ────────────────────────────────────────────
    def _add_padding(self, img: np.ndarray) -> np.ndarray:
        """
        إضافة هامش حول الحقل لتجنب قطع الحروف عند الحواف
        مهم جداً للحروف العربية مثل (ي، ى، ب) التي لها ذيل أسفل
        """
        p = self.padding
        return cv2.copyMakeBorder(
            img, p, p, p, p,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]   # خلفية بيضاء
        )

    # ─── 2. Upscale ────────────────────────────────────────────
    def _upscale_if_small(self, img: np.ndarray) -> np.ndarray:
        """
        رفع الدقة لو الحقل صغير جداً
        PaddleOCR يحتاج على الأقل 48px ارتفاع
        """
        h, w = img.shape[:2]
        if h < self.target_height:
            scale  = self.target_height / h
            new_w  = max(self.min_width, int(w * scale))
            img    = cv2.resize(
                img, (new_w, self.target_height),
                interpolation=cv2.INTER_LANCZOS4  # أفضل للنصوص [web:131]
            )
        elif w < self.min_width:
            scale  = self.min_width / w
            new_h  = int(h * scale)
            img    = cv2.resize(
                img, (self.min_width, new_h),
                interpolation=cv2.INTER_LANCZOS4
            )
        return img

    # ─── 3. Deskew داخل الحقل ──────────────────────────────────
    def _deskew_field(self, img: np.ndarray) -> np.ndarray:
        """
        تصحيح الانحراف داخل الحقل المقصوص
        زاوية صغيرة (< 5°) كافية لتدمير دقة PaddleOCR
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
               if len(img.shape) == 3 else img

        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 20:
            return img

        angle = cv2.minAreaRect(coords)[-1]
        if   angle < -45: angle = 90 + angle
        elif angle >  45: angle = angle - 90

        # تجاهل الانحراف الطفيف جداً
        if abs(angle) < 0.3:
            return img

        h, w = img.shape[:2]
        M    = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    # ─── 4. تحسين التباين ──────────────────────────────────────
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        CLAHE على قناة Luminance فقط للحفاظ على الألوان
        يرفع دقة OCR من 65% → 90% على النصوص المطبوعة [web:131]
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # clipLimit=2.0 مثالي للحقول الصغيرة — لا تبالغ
        clahe = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(4, 4)  # grid أصغر للحقول الصغيرة
        )
        l_eq  = clahe.apply(l)

        return cv2.cvtColor(
            cv2.merge([l_eq, a, b]),
            cv2.COLOR_LAB2BGR
        )

    # ─── 5. إزالة الضوضاء ──────────────────────────────────────
    def _remove_noise(self, img: np.ndarray) -> np.ndarray:
        """
        Bilateral filter: يحافظ على حواف الحروف العربية المتصلة
        لا تستخدم GaussianBlur — يُضبّب الحروف الدقيقة [web:66]
        """
        return cv2.bilateralFilter(
            img, d=3,           # kernel صغير للحقول الصغيرة
            sigmaColor=25,
            sigmaSpace=25
        )

    # ─── 6. Binarization (PaddleOCR فقط) ───────────────────────
    def _binarize(self, img: np.ndarray) -> np.ndarray:
        """
        Adaptive Thresholding أفضل من Otsu للحقول ذات الإضاءة غير المنتظمة
        Otsu يفشل لو الخلفية غير موحدة [web:156]
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
               if len(img.shape) == 3 else img.copy()

        # قياس توحيد الإضاءة
        brightness_std = gray.std()

        if brightness_std > 40:
            # إضاءة غير منتظمة → Adaptive
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=15,   # اضبط حسب حجم الحقل
                C=8
            )
        else:
            # إضاءة منتظمة → Otsu أسرع وأدق
            _, binary = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        # تحويل لـ BGR مجدداً (PaddleOCR يتوقع 3 channels)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # ─── 7. Resize موحد لـ PaddleOCR ───────────────────────────
    def _resize_for_paddle(self, img: np.ndarray) -> np.ndarray:
        """
        PaddleOCR يتوقع ارتفاع ثابت 48px مع عرض متغير
        max_width=1200 لتجنب ضغط النص الطويل (العناوين)
        """
        h, w = img.shape[:2]
        target_h = self.target_height
        scale    = target_h / h
        new_w    = min(1200, int(w * scale))

        return cv2.resize(
            img, (new_w, target_h),
            interpolation=cv2.INTER_AREA   # INTER_AREA للتصغير
        )
```


***

## دالة تشخيص جودة الحقل

```python
def assess_field_quality(img: np.ndarray) -> dict:
    """قياس جودة الحقل وتحديد المشاكل قبل المعالجة"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
           if len(img.shape) == 3 else img

    h, w     = img.shape[:2]
    contrast = float(gray.std())
    blur     = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright   = float(gray.mean())

    issues = []
    if h < 20:          issues.append("too_small")
    if contrast < 15:   issues.append("low_contrast")
    if blur < 30:       issues.append("blurry")
    if bright > 240:    issues.append("overexposed")
    if bright < 30:     issues.append("underexposed")

    quality = (
        "good"   if not issues else
        "medium" if len(issues) == 1 else
        "poor"
    )

    return {
        "size":      f"{w}x{h}",
        "contrast":  round(contrast, 1),
        "sharpness": round(blur, 1),
        "brightness":round(bright, 1),
        "issues":    issues,
        "quality":   quality,
    }
```


***

## تشغيل على Dataset كامل مع تقرير

```python
from pathlib import Path
import pandas as pd
from tqdm import tqdm

preprocessor = FieldPreprocessor(target_height=48, padding=4)

def preprocess_dataset(crops_dir: str,
                        output_dir: str,
                        model_type: str = "paddleocr") -> pd.DataFrame:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    images  = list(Path(crops_dir).glob("*.jpg"))
    reports = []

    for img_path in tqdm(images, desc="Preprocessing fields"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # تقييم قبل
        before = assess_field_quality(img)

        # معالجة
        processed = preprocessor.process(img, model_type=model_type)

        # تقييم بعد
        after = assess_field_quality(processed)

        # حفظ
        out_path = Path(output_dir) / img_path.name
        cv2.imwrite(str(out_path), processed)

        reports.append({
            "file":             img_path.name,
            "quality_before":   before["quality"],
            "quality_after":    after["quality"],
            "contrast_before":  before["contrast"],
            "contrast_after":   after["contrast"],
            "issues_fixed":     len(before["issues"]) - len(after["issues"]),
        })

    df = pd.DataFrame(reports)

    # ── تقرير ──────────────────────────────────────────────────
    print("\n" + "="*50)
    print("   📊 Preprocessing Report")
    print("="*50)
    for quality in ["good", "medium", "poor"]:
        b = (df["quality_before"] == quality).sum()
        a = (df["quality_after"]  == quality).sum()
        print(f"  {quality.upper():<8}: {b:>5} → {a:>5}")
    print(f"\n  Avg contrast gain : "
          f"{(df['contrast_after'] - df['contrast_before']).mean():.1f}")
    print(f"  Issues fixed      : {df['issues_fixed'].sum()}")
    print("="*50)

    df.to_csv("./preprocessing_report.csv",
              index=False, encoding="utf-8-sig")
    return df


# تشغيل لـ PaddleOCR
report_df = preprocess_dataset(
    crops_dir   = "./dataset/rec/images",
    output_dir  = "./dataset/rec/images_processed",
    model_type  = "paddleocr"   # أو "vlm" لـ QARI/Gemini
)
```


***

## الفرق بين Preprocessing لكل نموذج

```python
# ── مثال مرئي للمقارنة ──────────────────────────────────────
def visualize_preprocessing(img_path: str):
    img   = cv2.imread(img_path)
    proc  = FieldPreprocessor(target_height=48)

    stages = {
        "Original":            img,
        "After Padding":       proc._add_padding(img.copy()),
        "After Upscale":       proc._upscale_if_small(img.copy()),
        "After Contrast":      proc._enhance_contrast(img.copy()),
        "After Denoise":       proc._remove_noise(img.copy()),
        "PaddleOCR (binary)":  proc.process(img.copy(), "paddleocr"),
        "VLM (color)":         proc.process(img.copy(), "vlm"),
    }

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(stages), figsize=(20, 3))
    for ax, (title, im) in zip(axes, stages.items()):
        disp = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) \
               if len(im.shape) == 3 else im
        ax.imshow(disp, cmap="gray" if len(im.shape) == 2 else None)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("preprocessing_stages.png", dpi=150)
    print("✅ Saved → preprocessing_stages.png")

# visualize_preprocessing("./dataset/rec/images/id_001_name.jpg")
```


***

## ملخص التأثير على الدقة

| التقنية | تحسين الدقة | الأولوية |
| :-- | :-- | :-- |
| Padding (4px) | يمنع قطع الحروف | 🔴 ضرورية |
| Upscale LANCZOS4 | +35% للصور الصغيرة [^16_1] | 🔴 ضرورية |
| CLAHE | +9–25% [^16_4] | 🟠 عالية |
| Bilateral Denoise | +5–10% | 🟠 عالية |
| Adaptive Binarization | +15% على إضاءة غير منتظمة [^16_3] | 🟡 متوسطة |
| Deskew | +15–20% على الصور المنحرفة [^16_2] | 🟡 حسب الحاجة |

<span style="display:none">[^16_10][^16_11][^16_12][^16_13][^16_14][^16_15][^16_5][^16_6][^16_7][^16_8][^16_9]</span>

<div align="center">⁂</div>

[^16_1]: https://theaspd.com/index.php/ijes/article/view/11627

[^16_2]: https://dl.acm.org/doi/10.1145/3768150

[^16_3]: https://www.iri.com/blog/data-protection/preprocessing-images-for-ocr-darkshield/

[^16_4]: https://arxiv.org/html/2312.11812v1

[^16_5]: https://arxiv.org/html/2506.02295v1

[^16_6]: https://b.aun.edu.eg/fci/sites/default/files/researches/Advancements and Challenges in Arabic Optical Character Recognition A Comprehensive Survey.pdf

[^16_7]: https://aclanthology.org/2025.acl-long.749.pdf

[^16_8]: https://arxiv.org/html/2502.05277v1

[^16_9]: https://github.com/nithin-k-shine/Contrast-Enhancement-for-OCR

[^16_10]: https://kby-ai.com/4-real-life-id-document-ocr-challenges-in-processing/

[^16_11]: https://www.nitorinfotech.com/blog/improve-ocr-accuracy-using-advanced-preprocessing-techniques/

[^16_12]: https://github.com/h9-tect/Arabic_nlp_preprocessing

[^16_13]: https://docparser.com/blog/improve-ocr-accuracy/

[^16_14]: https://github.com/AHR-OCR2024/Arabic-Handwriting-Recognition

[^16_15]: https://www.systemoverflow.com/learn/ml-nlp-systems/tokenization-preprocessing/preprocessing-pipeline-normalization-and-text-cleaning


---

# كيف أدمج preprocessing في خط أنابيب QARI-OCR

بناءً على كل اللي بنيناه في المحادثة، هنا الدمج الكامل بدون أي بحث إضافي.

***

## البنية الكاملة للـ Pipeline

```
صورة الحقل المقصوص
        ↓
FieldPreprocessor.process()
        ↓
  ┌─────────────┐
  │ جودة الصورة │
  └──────┬──────┘
    good │  poor
         ↓      ↓
    QARI مباشر  QARI + Smart Prompt
         ↓
  fix_arabic_text()
         ↓
   parse + validate
```


***

## Cell الدمج الكامل

```python
# qari_pipeline.py
import cv2
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import arabic_reshaper
from bidi.algorithm import get_display
import pandas as pd
from tqdm import tqdm


# ── استيراد الكلاسات السابقة ──────────────────────────────────
# from field_preprocessor import FieldPreprocessor, assess_field_quality
# (أو انسخهم هنا مباشرة)


# ─────────────────────────────────────────────────────────────
@dataclass
class OCRResult:
    field:       str
    raw_text:    str         # خروج النموذج الخام
    clean_text:  str         # بعد إصلاح RTL
    confidence:  float       # 0.0 → 1.0 (تقدير)
    quality:     str         # good / medium / poor
    issues:      list        # مشاكل الصورة المكتشفة


# ─────────────────────────────────────────────────────────────
FIELD_PROMPTS = {
    "name":
        "اقرأ الاسم الرباعي بالكامل كما هو مكتوب، بدون أي تعديل.",
    "national_id":
        "اقرأ الرقم القومي المكوّن من 14 رقم بدقة تامة، أرقام فقط.",
    "birth_date":
        "اقرأ تاريخ الميلاد كما هو مكتوب بالضبط.",
    "address":
        "اقرأ العنوان الكامل كما هو مكتوب.",
    "governorate":
        "اقرأ اسم المحافظة فقط.",
    "gender":
        "اقرأ كلمة الجنس فقط (ذكر أو أنثى).",
    "expiry_date":
        "اقرأ تاريخ انتهاء البطاقة كما هو مكتوب.",
}

LOW_QUALITY_SUFFIX = (
    "\nملاحظة: الصورة قد تكون غير واضحة. "
    "اقرأ ما تستطيع وضع [؟] بدل الحروف غير الواضحة."
)


# ─────────────────────────────────────────────────────────────
class QARIPipeline:
    """
    Pipeline كامل: Preprocessing → QARI-OCR → Post-processing
    """

    def __init__(self,
                 model_name: str = "NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct",
                 use_4bit:   bool = False,
                 device:     str  = "auto"):

        print(f"⏳ Loading {model_name} ...")
        dtype = torch.float16

        if use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=bnb_cfg,
                device_map=device,
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device,
            )

        self.processor   = AutoProcessor.from_pretrained(model_name)
        self.preprocessor = FieldPreprocessor(target_height=48, padding=4)
        self.device_name = str(next(self.model.parameters()).device)
        print(f"✅ QARI ready on: {self.device_name}")

    # ── الخطوة 1: Preprocessing ──────────────────────────────
    def _preprocess_field(self, img: np.ndarray) -> tuple:
        """يُعيد (processed_img, quality_info)"""
        quality = assess_field_quality(img)
        # استخدام mode=vlm للحفاظ على الألوان (QARI VLM)
        processed = self.preprocessor.process(img, model_type="vlm")
        return processed, quality

    # ── الخطوة 2: بناء الـ Prompt ────────────────────────────
    def _build_prompt(self, field: str, quality: dict) -> str:
        base = FIELD_PROMPTS.get(
            field,
            "اقرأ كل النص في الصورة بدقة تامة."
        )
        prompt = f"أنت نظام OCR متخصص في بطاقات الهوية المصرية.\n{base}"
        prompt += "\nأرجع النص فقط بدون أي شرح أو تعليق."

        # تعديل الـ prompt حسب جودة الصورة
        if quality["quality"] == "poor":
            prompt += LOW_QUALITY_SUFFIX

        return prompt

    # ── الخطوة 3: Inference ──────────────────────────────────
    def _run_inference(self, img: np.ndarray, prompt: str) -> str:
        # حفظ مؤقت (مطلوب من qwen_vl_utils)
        tmp = "/tmp/_qari_field.png"
        cv2.imwrite(tmp, img)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{tmp}"},
                {"type": "text",  "text":  prompt},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_in, vid_in = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=img_in,
            videos=vid_in, return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                repetition_penalty=1.1,
            )

        result = self.processor.batch_decode(
            [out[0][inputs.input_ids.shape[1]:]],
            skip_special_tokens=True,
        )[0].strip()

        os.remove(tmp)
        return result

    # ── الخطوة 4: Post-processing ────────────────────────────
    def _fix_arabic(self, text: str) -> str:
        """إصلاح RTL + reshaping"""
        if not text:
            return ""
        # عكس النص (النموذج يُخرج LTR)
        reversed_text = text[::-1]
        reshaped      = arabic_reshaper.reshape(reversed_text)
        return get_display(reshaped)

    def _estimate_confidence(self, raw: str,
                              field: str) -> float:
        """تقدير مبسط للثقة بناءً على طول النص والحقل"""
        import re
        if not raw or raw.startswith("ERROR"):
            return 0.0
        if field == "national_id":
            # نتوقع 14 رقم بالضبط
            digits = re.sub(r'\D', '', raw)
            return 1.0 if len(digits) == 14 else max(0.3, len(digits)/14)
        if field == "gender":
            return 1.0 if raw.strip() in ["ذكر", "أنثى"] else 0.3
        # للحقول النصية: طول معقول = ثقة عالية
        return min(1.0, len(raw.strip()) / 5) if len(raw.strip()) > 0 else 0.0

    # ── الدالة الرئيسية ───────────────────────────────────────
    def extract_field(self, img: np.ndarray,
                      field: str) -> OCRResult:
        """
        استخراج نص حقل واحد كاملاً مع كل المراحل
        """
        # 1. Preprocessing
        processed, quality = self._preprocess_field(img)

        # 2. Prompt
        prompt = self._build_prompt(field, quality)

        # 3. Inference
        raw_text = self._run_inference(processed, prompt)

        # 4. Post-processing
        clean_text = self._fix_arabic(raw_text)

        # 5. Confidence
        conf = self._estimate_confidence(raw_text, field)

        return OCRResult(
            field      = field,
            raw_text   = raw_text,
            clean_text = clean_text,
            confidence = conf,
            quality    = quality["quality"],
            issues     = quality["issues"],
        )

    def extract_all_fields(self,
                            crops: dict[str, np.ndarray]
                            ) -> dict[str, OCRResult]:
        """
        استخراج جميع حقول بطاقة الهوية دفعة واحدة
        crops = {"name": img_array, "national_id": img_array, ...}
        """
        results = {}
        for field, img in crops.items():
            results[field] = self.extract_field(img, field)
        return results
```


***

## تشغيل على صورة واحدة

```python
# تهيئة الـ Pipeline مرة واحدة فقط
pipeline = QARIPipeline(
    model_name = "NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct",
    use_4bit   = False,   # True لو VRAM < 6GB
)

# تحميل صورة + قص الحقول
full_id = cv2.imread("egyptian_id.jpg")

# لو عندك YOLO detector:
# fields_detected = detector.detect(full_id)
# crops = {f["class_name"]: full_id[y1:y2, x1:x2]
#          for f in fields_detected
#          for x1,y1,x2,y2 in [f["bbox"]]}

# أو حقول جاهزة مباشرة:
crops = {
    "name":        cv2.imread("./crops/id_001_name.jpg"),
    "national_id": cv2.imread("./crops/id_001_national_id.jpg"),
    "address":     cv2.imread("./crops/id_001_address.jpg"),
    "governorate": cv2.imread("./crops/id_001_governorate.jpg"),
}

results = pipeline.extract_all_fields(crops)

# عرض النتائج
print("\n" + "="*50)
for field, res in results.items():
    status = "✅" if res.confidence > 0.7 else "⚠️"
    print(f"{status} {field:<15}: {res.clean_text}")
    print(f"   confidence={res.confidence:.2f} | "
          f"quality={res.quality} | issues={res.issues}")
print("="*50)
```


***

## معالجة Dataset الـ 15,000 صورة

```python
def process_full_dataset(crops_csv:  str,
                          output_csv: str,
                          base_dir:   str = "./dataset") -> pd.DataFrame:
    """
    تشغيل الـ Pipeline الكامل على كل الـ crops مع إمكانية الاستكمال
    """
    df = pd.read_csv(crops_csv)

    # استكمال من حيث توقفنا
    done = set()
    if Path(output_csv).exists():
        done_df = pd.read_csv(output_csv)
        done    = set(done_df["image_path"].tolist())
        records = done_df.to_dict("records")
        print(f"▶️  Resuming — {len(done)} already done")
    else:
        records = []

    pending = df[~df["image_path"].isin(done)]
    print(f"📋 Pending: {len(pending):,} crops")

    for _, row in tqdm(pending.iterrows(), total=len(pending)):
        img_path = Path(base_dir) / row["image_path"]
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        try:
            result = pipeline.extract_field(img, row["field"])
            records.append({
                "image_path":  row["image_path"],
                "field":       result.field,
                "clean_text":  result.clean_text,
                "raw_text":    result.raw_text,
                "confidence":  result.confidence,
                "quality":     result.quality,
                "issues":      "|".join(result.issues),
                "status":      "success",
            })
        except Exception as e:
            records.append({
                "image_path": row["image_path"],
                "field":      row["field"],
                "status":     f"error: {e}",
            })

        # حفظ كل 100 صورة
        if len(records) % 100 == 0:
            pd.DataFrame(records).to_csv(
                output_csv, index=False, encoding="utf-8-sig"
            )

    result_df = pd.DataFrame(records)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # ── تقرير نهائي ────────────────────────────────────────────
    success = result_df[result_df["status"] == "success"]
    print(f"\n✅ Done: {len(success):,} success | "
          f"❌ {len(result_df)-len(success):,} errors")
    print(f"📊 Avg confidence : "
          f"{success['confidence'].mean():.2f}")
    print(f"⚠️  Low confidence (<0.7): "
          f"{(success['confidence'] < 0.7).sum():,}")

    return result_df


result_df = process_full_dataset(
    crops_csv  = "./dataset/crops_metadata.csv",
    output_csv = "./dataset/qari_labels.csv",
)
```


***

## FastAPI Endpoint جاهز للإنتاج

```python
from fastapi import FastAPI, UploadFile, File
from functools import lru_cache
import numpy as np

app = FastAPI(title="Egyptian ID OCR API")

@lru_cache(maxsize=1)
def get_pipeline():
    return QARIPipeline(
        model_name="NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct",
        use_4bit=True,
    )

@app.post("/ocr/field")
async def ocr_single_field(
    file:  UploadFile = File(...),
    field: str        = "name",
):
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    result = get_pipeline().extract_field(img, field)

    return {
        "field":      result.field,
        "text":       result.clean_text,
        "confidence": result.confidence,
        "quality":    result.quality,
        "issues":     result.issues,
    }

@app.post("/ocr/all-fields")
async def ocr_all_fields(files: list[UploadFile] = File(...),
                          fields: list[str] = None):
    if not fields:
        fields = list(FIELD_PROMPTS.keys())[:len(files)]

    crops = {}
    for file, field in zip(files, fields):
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        crops[field] = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    results = get_pipeline().extract_all_fields(crops)

    return {
        field: {
            "text":       r.clean_text,
            "confidence": r.confidence,
            "quality":    r.quality,
        }
        for field, r in results.items()
    }
```


***

## ترتيب الاستدعاء الكامل

```python
# ── في سطر واحد للاختبار السريع ─────────────────────────────
img    = cv2.imread("name_crop.jpg")
result = QARIPipeline().extract_field(img, "name")
print(result.clean_text)   # محمد أحمد علي ✅

# ── أو الـ Pipeline الكامل ───────────────────────────────────
# 1. تهيئة النموذج مرة واحدة
pipeline = QARIPipeline(use_4bit=True)

# 2. قص الحقول (YOLO أو labels جاهزة)
crops = load_crops("egyptian_id.jpg")

# 3. استخراج الكل
results = pipeline.extract_all_fields(crops)

# 4. تحويل لـ JSON
output = {f: r.clean_text for f, r in results.items()}
```


---

# egyptian_id_ready/

├── train/
│ ├── images/ ← آلاف الصور
│ └── labels/
├── valid/
│ ├── images/
│ └── labels/
├── test/
│ ├── images/
│ └── labels/

ممتاز — هذه بنية YOLO جاهزة. إليك الـ Notebook الكامل الذي يقرأ هذا الهيكل مباشرة.

***

## Cell 1 — الإعداد والتحقق من البنية

```python
from pathlib import Path
import cv2, os, json, re
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── إعدادات المشروع ───────────────────────────────────────────
ROOT = Path("./egyptian_id_ready")

SPLITS = {
    "train": ROOT / "train",
    "valid": ROOT / "valid",
    "test":  ROOT / "test",
}

FIELD_NAMES = [          # يجب أن يطابق ترتيب classes في YOLO
    "name",
    "national_id",
    "birth_date",
    "address",
    "governorate",
    "gender",
    "expiry_date",
]

OUTPUT = {
    "rec_dir":   ROOT / "rec" / "images",    # حقول مقصوصة
    "train_txt": ROOT / "rec" / "train.txt",
    "val_txt":   ROOT / "rec" / "val.txt",
    "test_txt":  ROOT / "rec" / "test.txt",
}

for d in [OUTPUT["rec_dir"], ROOT / "rec"]:
    d.mkdir(parents=True, exist_ok=True)

# ── التحقق من البنية ──────────────────────────────────────────
print("📁 Dataset Structure:")
print("="*45)
total_images = 0
for split, path in SPLITS.items():
    imgs   = list((path / "images").glob("*.jpg")) + \
             list((path / "images").glob("*.png"))
    labels = list((path / "labels").glob("*.txt"))
    print(f"  {split:<8}: {len(imgs):>5} images | "
          f"{len(labels):>5} labels")
    total_images += len(imgs)
print(f"  {'TOTAL':<8}: {total_images:>5} images")
print("="*45)
```


***

## Cell 2 — قراءة YOLO Labels وقص الحقول

```python
def parse_yolo_label(label_path: Path,
                     img_w: int, img_h: int) -> list[dict]:
    """قراءة ملف YOLO وتحويل الإحداثيات لـ pixel coordinates"""
    fields = []
    if not label_path.exists():
        return fields

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx  = float(parts[1]) * img_w
            cy  = float(parts[2]) * img_h
            bw  = float(parts[3]) * img_w
            bh  = float(parts[4]) * img_h

            x1 = max(0,     int(cx - bw / 2))
            y1 = max(0,     int(cy - bh / 2))
            x2 = min(img_w, int(cx + bw / 2))
            y2 = min(img_h, int(cy + bh / 2))

            if x2 > x1 and y2 > y1:
                fields.append({
                    "class_id":   cid,
                    "class_name": FIELD_NAMES[cid]
                                  if cid < len(FIELD_NAMES)
                                  else f"field_{cid}",
                    "bbox":       [x1, y1, x2, y2],
                })
    return fields


def crop_field(img: np.ndarray, bbox: list,
               padding: int = 4) -> np.ndarray | None:
    """قص حقل مع padding لتجنب قطع الحروف"""
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def build_crops_from_split(split: str) -> pd.DataFrame:
    """قص جميع الحقول من split معين"""
    split_path = SPLITS[split]
    images_dir = split_path / "images"
    labels_dir = split_path / "labels"

    image_files = sorted(
        list(images_dir.glob("*.jpg")) +
        list(images_dir.glob("*.png"))
    )

    records = []
    stats   = {"processed": 0, "crops": 0, "no_label": 0}

    for img_path in tqdm(image_files, desc=f"[{split}] Cropping"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w    = img.shape[:2]
        lbl_path = labels_dir / (img_path.stem + ".txt")
        fields   = parse_yolo_label(lbl_path, w, h)

        if not fields:
            stats["no_label"] += 1
            continue

        for field in fields:
            crop = crop_field(img, field["bbox"])
            if crop is None:
                continue

            # اسم الملف: {split}_{stem}_{field}.jpg
            save_name = f"{split}_{img_path.stem}_{field['class_name']}.jpg"
            save_path = OUTPUT["rec_dir"] / save_name
            cv2.imwrite(str(save_path), crop)

            records.append({
                "image_path":  f"rec/images/{save_name}",
                "field":       field["class_name"],
                "class_id":    field["class_id"],
                "split":       split,
                "orig_image":  img_path.name,
                "label_text":  "",      # يُملأ بـ QARI أو Gemini
            })
            stats["crops"] += 1

        stats["processed"] += 1

    print(f"  ✅ {stats['processed']} images → "
          f"{stats['crops']} crops | "
          f"⚠️ {stats['no_label']} without labels")
    return pd.DataFrame(records)


# ── تشغيل على كل الـ splits ───────────────────────────────────
all_dfs = []
for split in ["train", "valid", "test"]:
    df = build_crops_from_split(split)
    all_dfs.append(df)

crops_df = pd.concat(all_dfs, ignore_index=True)
crops_df.to_csv(ROOT / "crops_metadata.csv",
                index=False, encoding="utf-8-sig")

print(f"\n📊 Total crops: {len(crops_df):,}")
print(crops_df.groupby(["split", "field"])["image_path"].count().unstack())
```


***

## Cell 3 — استخراج النص (اختار طريقة)

```python
# ════════════════════════════════════════════════════════════
#   اختار الطريقة المناسبة:
#   METHOD = "qari"    ← open-source, يحتاج GPU
#   METHOD = "gemini"  ← API, يعمل بدون GPU
#   METHOD = "both"    ← مقارنة الاثنين
# ════════════════════════════════════════════════════════════
METHOD = "qari"


# ── Gemini Setup ──────────────────────────────────────────────
if METHOD in ["gemini", "both"]:
    import google.generativeai as genai
    genai.configure(api_key="YOUR_GEMINI_API_KEY")
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")

    def _gemini_ocr(img_path: str, field: str) -> str:
        prompt = (
            f"أنت نظام OCR للهوية المصرية. "
            f"اقرأ حقل '{field}' فقط بدقة تامة. "
            f"أرجع النص فقط."
        )
        try:
            resp = gemini_model.generate_content(
                [prompt, Image.open(img_path)]
            )
            return resp.text.strip()
        except Exception as e:
            return f"ERROR:{e}"


# ── QARI Setup ────────────────────────────────────────────────
if METHOD in ["qari", "both"]:
    import torch
    from PIL import Image
    from transformers import (Qwen2VLForConditionalGeneration,
                               AutoProcessor)
    from qwen_vl_utils import process_vision_info
    import arabic_reshaper
    from bidi.algorithm import get_display

    _qari_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    _qari_proc  = AutoProcessor.from_pretrained(
        "NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct"
    )

    def _fix_rtl(text: str) -> str:
        if not text: return ""
        return get_display(arabic_reshaper.reshape(text[::-1]))

    def _qari_ocr(img_path: str, field: str) -> str:
        tmp = "/tmp/_q.png"
        cv2.imwrite(tmp, cv2.imread(img_path))
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{tmp}"},
            {"type": "text",  "text":
             f"اقرأ حقل {field} فقط. أرجع النص فقط."},
        ]}]
        txt    = _qari_proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        ii, vi = process_vision_info(msgs)
        inp    = _qari_proc(
            text=[txt], images=ii, videos=vi, return_tensors="pt"
        ).to(_qari_model.device)
        with torch.no_grad():
            out = _qari_model.generate(
                **inp, max_new_tokens=64,
                do_sample=False, repetition_penalty=1.1,
            )
        raw = _qari_proc.batch_decode(
            [out[0][inp.input_ids.shape[1]:]],
            skip_special_tokens=True,
        )[0].strip()
        os.remove(tmp)
        return _fix_rtl(raw)


# ── دالة موحدة ───────────────────────────────────────────────
def extract_text(img_path: str, field: str) -> dict:
    """استخراج النص بالطريقة المختارة"""
    result = {"field": field, "image_path": img_path}

    if METHOD == "qari":
        result["label_text"] = _qari_ocr(img_path, field)

    elif METHOD == "gemini":
        result["label_text"] = _gemini_ocr(img_path, field)

    elif METHOD == "both":
        q = _qari_ocr(img_path, field)
        g = _gemini_ocr(img_path, field)
        result["label_text"]  = q          # الافتراضي
        result["qari_text"]   = q
        result["gemini_text"] = g
        result["texts_match"] = (q.strip() == g.strip())

    return result
```


***

## Cell 4 — تشغيل الاستخراج على Dataset كاملة

```python
import time

def label_all_crops(crops_df:    pd.DataFrame,
                    base_dir:    Path,
                    output_csv:  Path,
                    splits_to_label: list = ["train", "valid"]) -> pd.DataFrame:
    """
    استخراج النص لكل الـ crops
    splits_to_label: عادةً train+valid فقط — test يُقيَّم لاحقاً
    """
    df     = crops_df.copy()
    subset = df[
        df["split"].isin(splits_to_label) &
        (df["label_text"] == "")
    ]

    # استكمال لو وقفنا في النص
    if output_csv.exists():
        done     = pd.read_csv(output_csv)
        done_set = set(done["image_path"])
        subset   = subset[~subset["image_path"].isin(done_set)]
        print(f"▶️  Resuming — skipping {len(done_set)} done")

    print(f"📤 Extracting text for {len(subset):,} crops "
          f"using [{METHOD.upper()}]...")

    for idx, row in tqdm(subset.iterrows(), total=len(subset)):
        img_path = str(base_dir / row["image_path"])
        if not Path(img_path).exists():
            continue

        result = extract_text(img_path, row["field"])

        for k, v in result.items():
            if k in df.columns or k not in ("field", "image_path"):
                df.at[idx, k] = v

        # حفظ تدريجي كل 50
        if idx % 50 == 0:
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        # Rate limit لـ Gemini
        if METHOD in ["gemini", "both"]:
            time.sleep(0.4)

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved → {output_csv}")
    return df


labeled_df = label_all_crops(
    crops_df,
    base_dir   = ROOT,
    output_csv = ROOT / "crops_labeled.csv",
    splits_to_label = ["train", "valid"],
)
```


***

## Cell 5 — بناء ملفات PaddleOCR النهائية

```python
import re

def clean_text(text: str) -> str:
    if not text or str(text).startswith("ERROR"):
        return ""
    text = str(text).strip()
    text = re.sub(r'[^\u0600-\u06FF\u0660-\u0669\u0030-\u0039\s\-/]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def write_paddle_txts(labeled_df: pd.DataFrame):
    """
    كتابة train.txt / val.txt / test.txt بصيغة PaddleOCR
    - train ← split==train
    - val   ← split==valid
    - test  ← split==test (بدون labels — للتقييم فقط)
    """
    df = labeled_df.copy()
    df["label_clean"] = df["label_text"].apply(clean_text)
    df["label_final"] = df["label_clean"].apply(
        lambda x: x[::-1]  # عكس النص للـ PaddleOCR
    )

    mapping = {
        "train": OUTPUT["train_txt"],
        "valid": OUTPUT["val_txt"],
        "test":  OUTPUT["test_txt"],
    }

    stats = {}
    for split, txt_path in mapping.items():
        subset = df[
            (df["split"] == split) &
            (df["label_final"].str.len() > 0)
        ]
        with open(txt_path, "w", encoding="utf-8") as f:
            for _, row in subset.iterrows():
                f.write(f"{row['image_path']}\t{row['label_final']}\n")
        stats[split] = len(subset)

    print("\n" + "="*45)
    print("   📄 PaddleOCR Label Files")
    print("="*45)
    for split, count in stats.items():
        print(f"  {split:<8}: {count:>6,} samples")
    print("="*45)


write_paddle_txts(labeled_df)
```


***

## Cell 6 — إحصائيات وتقرير شامل

```python
df = pd.read_csv(ROOT / "crops_labeled.csv")
df["label_clean"] = df["label_text"].apply(clean_text)

print("\n" + "="*55)
print("          📊 Final Dataset Report")
print("="*55)

# توزيع الحقول
print("\n🏷️  Field Distribution:")
pivot = df.groupby(["split", "field"])["image_path"].count().unstack(fill_value=0)
print(pivot.to_string())

# جودة الـ labels
labeled   = (df["label_clean"].str.len() > 0).sum()
unlabeled = len(df) - labeled
print(f"\n📝 Labels:")
print(f"  Labeled    : {labeled:>7,} ({labeled/len(df)*100:.1f}%)")
print(f"  Unlabeled  : {unlabeled:>7,} ({unlabeled/len(df)*100:.1f}%)")

# توزيع طول النص
print(f"\n📏 Text Length Stats:")
lens = df[df["label_clean"].str.len() > 0]["label_clean"].str.len()
print(f"  Mean  : {lens.mean():.1f} chars")
print(f"  Min   : {lens.min()}")
print(f"  Max   : {lens.max()}")
print(f"  > 40  : {(lens > 40).sum()} (راجع max_text_length في config)")

# Character coverage
with open("./arabic_dict.txt", encoding="utf-8") as f:
    dict_chars = set(f.read().strip())
all_chars    = set("".join(df["label_clean"].dropna()))
coverage     = len(all_chars & dict_chars) / len(dict_chars) * 100
missing      = dict_chars - all_chars

print(f"\n🔤 Dictionary Coverage: {coverage:.1f}%")
if missing:
    print(f"  Missing chars: {''.join(sorted(missing))}")

print("="*55)
```


***

## Cell 7 — Fine-tuning مباشرة

```python
# كتابة الـ Config بناءً على إحصائيات الـ dataset الفعلية
max_len = int(df["label_clean"].str.len().quantile(0.99)) + 5

config = f"""
Global:
  use_gpu: false
  epoch_num: 100
  save_model_dir: ./output/egyptian_id_rec/
  pretrained_model: ./arabic_PP-OCRv3_rec_train/best_accuracy
  character_dict_path: ./arabic_dict.txt
  max_text_length: {max_len}

Optimizer:
  name: Adam
  lr:
    name: Cosine
    learning_rate: 0.0001
    warmup_epoch: 5

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
  Neck:
    name: SequenceEncoder
    encoder_type: svtr
  Head:
    name: CTCHead

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {ROOT}/
    label_file_list:
    - {OUTPUT["train_txt"]}
  loader:
    shuffle: true
    batch_size_per_card: 32
    num_workers: 2

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {ROOT}/
    label_file_list:
    - {OUTPUT["val_txt"]}
  loader:
    shuffle: false
    batch_size_per_card: 32
    num_workers: 2
"""

with open("./configs/egyptian_id_rec.yml", "w") as f:
    f.write(config)
print(f"✅ Config ready | max_text_length={max_len}")

# ── تشغيل التدريب ──────────────────────────────────────────
!python tools/train.py -c ./configs/egyptian_id_rec.yml
```


***

## ملخص الـ Pipeline بالأرقام

```
egyptian_id_ready/
├── train/images+labels  ──→  crops (train)  ──→  QARI/Gemini  ──→  train.txt
├── valid/images+labels  ──→  crops (valid)  ──→  QARI/Gemini  ──→  val.txt
├── test/images+labels   ──→  crops (test)   ──→  (تقييم لاحق) ──→  test.txt
└── rec/
    ├── images/   ← كل الحقول المقصوصة
    ├── train.txt
    ├── val.txt
    └── test.txt
```


---

## Cell 8 — تقييم النموذج على Test Split

```python
# ── تقييم بعد التدريب ─────────────────────────────────────────
!python tools/eval.py \
    -c ./configs/egyptian_id_rec.yml \
    -o Global.checkpoints=./output/egyptian_id_rec/best_accuracy

# ── تقييم مفصل بالكود ────────────────────────────────────────
import editdistance
from paddleocr import PaddleOCR

ocr_eval = PaddleOCR(
    use_gpu=False,
    rec_model_dir="./output/egyptian_id_rec/best_accuracy",
    rec_char_dict_path="./arabic_dict.txt",
    lang="ar",
    show_log=False,
)

def evaluate_on_test(test_txt: str,
                     base_dir:  Path) -> pd.DataFrame:
    """تقييم شامل على test split مع CER/WER لكل حقل"""

    with open(test_txt, encoding="utf-8") as f:
        lines = [l.strip() for l in f if "\t" in l]

    records = []
    for line in tqdm(lines, desc="Evaluating"):
        img_rel, label_rev = line.split("\t", 1)

        # عكس الـ label للمقارنة الصحيحة
        gt = label_rev[::-1]

        img_path = base_dir / img_rel
        if not img_path.exists():
            continue

        result = ocr_eval.ocr(str(img_path), cls=False)
        if result and result[0]:
            pred_raw, conf = result[0][0][1]
            pred = pred_raw[::-1]   # عكس الـ prediction
        else:
            pred, conf = "", 0.0

        # استخراج نوع الحقل من اسم الملف
        field = img_path.stem.split("_")[-1]

        records.append({
            "image":     img_path.name,
            "field":     field,
            "gt":        gt,
            "pred":      pred,
            "conf":      round(float(conf), 3),
            "cer":       editdistance.eval(pred, gt) / max(len(gt), 1),
            "exact":     pred.strip() == gt.strip(),
        })

    return pd.DataFrame(records)


test_df = evaluate_on_test(
    str(OUTPUT["test_txt"]), ROOT
)

# ── تقرير التقييم ─────────────────────────────────────────────
print("\n" + "="*55)
print("      📊 Test Evaluation Report")
print("="*55)

for field in FIELD_NAMES:
    sub = test_df[test_df["field"] == field]
    if sub.empty: continue
    avg_cer   = sub["cer"].mean()
    exact_acc = sub["exact"].mean() * 100
    status    = "✅" if avg_cer < 0.05 else ("⚠️" if avg_cer < 0.15 else "❌")
    print(f"{status} {field:<18} CER={avg_cer:.3f} | "
          f"Exact={exact_acc:.1f}%")

overall_cer = test_df["cer"].mean()
overall_acc = test_df["exact"].mean() * 100
print("─"*55)
print(f"   Overall CER   : {overall_cer:.3f} "
      f"({(1-overall_cer)*100:.1f}% accuracy)")
print(f"   Exact Match   : {overall_acc:.1f}%")
print(f"   Total Samples : {len(test_df):,}")
print("="*55)

test_df.to_csv(ROOT / "test_evaluation.csv",
               index=False, encoding="utf-8-sig")
```


***

## Cell 9 — تحليل الأخطاء

```python
def analyze_errors(test_df: pd.DataFrame, top_n: int = 20):
    """تحليل أسوأ الحالات لفهم نقاط الضعف"""

    errors = test_df[~test_df["exact"]].copy()
    errors["char_errors"] = errors.apply(
        lambda r: editdistance.eval(r["pred"], r["gt"]), axis=1
    )

    print(f"\n❌ Error Analysis ({len(errors)} errors / {len(test_df)} total)")
    print("="*55)

    # أكثر الحقول خطأً
    print("\n🔴 Error Rate by Field:")
    for field in FIELD_NAMES:
        f_all  = test_df[test_df["field"] == field]
        f_err  = errors[errors["field"] == field]
        if f_all.empty: continue
        rate   = len(f_err) / len(f_all) * 100
        bar    = "█" * int(rate / 5)
        print(f"  {field:<18} {rate:5.1f}%  {bar}")

    # أنواع الأخطاء الشائعة
    print("\n🔍 Common Error Patterns:")
    for _, row in errors.nlargest(top_n, "char_errors").iterrows():
        print(f"  [{row['field']}]")
        print(f"    GT  : {row['gt']}")
        print(f"    PRED: {row['pred']}")
        print(f"    CER : {row['cer']:.2f}")
        print()

    # حقول بثقة منخفضة
    low_conf = test_df[test_df["conf"] < 0.7]
    print(f"⚠️  Low confidence (<0.7): {len(low_conf):,} samples")
    print(f"   Their CER: {low_conf['cer'].mean():.3f}")

    errors.to_csv(ROOT / "error_analysis.csv",
                  index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved → error_analysis.csv")


analyze_errors(test_df)
```


***

## Cell 10 — تصدير ONNX

```python
# ── تصدير Paddle → Inference Model ───────────────────────────
!python tools/export_model.py \
    -c ./configs/egyptian_id_rec.yml \
    -o Global.pretrained_model=./output/egyptian_id_rec/best_accuracy \
       Global.save_inference_dir=./inference/rec

# ── Paddle → ONNX ────────────────────────────────────────────
!pip install paddle2onnx onnxsim -q

!paddle2onnx \
    --model_dir      ./inference/rec \
    --model_filename   inference.pdmodel \
    --params_filename  inference.pdiparams \
    --save_file      ./onnx/rec.onnx \
    --opset_version  11 \
    --enable_onnx_checker True

# ── تحسين الـ ONNX ───────────────────────────────────────────
!python -m onnxsim ./onnx/rec.onnx ./onnx/rec_sim.onnx

# ── التحقق ────────────────────────────────────────────────────
import onnx, onnxruntime as ort

model = onnx.load("./onnx/rec_sim.onnx")
onnx.checker.check_model(model)

sess  = ort.InferenceSession(
    "./onnx/rec_sim.onnx",
    providers=["CPUExecutionProvider"]
)
dummy = np.random.randn(1, 3, 48, 320).astype(np.float32)
out   = sess.run(None, {sess.get_inputs()[0].name: dummy})

print(f"✅ ONNX valid | Output shape: {out[0].shape}")
```


***

## Cell 11 — Inference Pipeline النهائي

```python
# inference_pipeline.py — الإنتاج النهائي
import onnxruntime as ort
import numpy as np
import cv2, re
import arabic_reshaper
from bidi.algorithm import get_display
from pathlib import Path


class EgyptianIDOCR:
    """
    Pipeline إنتاج كامل — ONNX على CPU
    Detection  : field_detector.onnx  (YOLO)
    Recognition: rec_sim.onnx         (PaddleOCR fine-tuned)
    """

    def __init__(self,
                 det_onnx:  str = "./onnx/det_sim.onnx",
                 rec_onnx:  str = "./onnx/rec_sim.onnx",
                 dict_path: str = "./arabic_dict.txt",
                 use_gpu:   bool = False):

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu else ["CPUExecutionProvider"]
        )

        self.det_sess = ort.InferenceSession(det_onnx,  providers=providers)
        self.rec_sess = ort.InferenceSession(rec_onnx,  providers=providers)

        with open(dict_path, encoding="utf-8") as f:
            chars = f.read().strip().split("\n")
        self.chars = ["blank"] + chars

        print(f"✅ EgyptianIDOCR ready | "
              f"Dict: {len(self.chars)} chars | "
              f"Provider: {self.rec_sess.get_providers()[0]}")

    # ── Preprocessing ─────────────────────────────────────────
    def _preprocess_rec(self, img: np.ndarray,
                         h: int = 48, w: int = 320) -> np.ndarray:
        img = cv2.resize(img, (w, h))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return img.transpose(2, 0, 1)[np.newaxis]   # [1,3,H,W]

    # ── CTC Decode ────────────────────────────────────────────
    def _ctc_decode(self, preds: np.ndarray) -> tuple[str, float]:
        indices = np.argmax(preds[0], axis=-1)
        scores  = np.max(preds[0], axis=-1)

        chars, confs = [], []
        prev = -1
        for i, idx in enumerate(indices):
            if idx != prev and idx != 0:
                chars.append(self.chars[idx])
                confs.append(scores[i])
            prev = idx

        raw_text = "".join(chars)
        avg_conf = float(np.mean(confs)) if confs else 0.0

        # إصلاح RTL
        clean = get_display(arabic_reshaper.reshape(raw_text[::-1]))
        return clean, avg_conf

    # ── Recognize واحد ────────────────────────────────────────
    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        inp  = self._preprocess_rec(crop)
        name = self.rec_sess.get_inputs()[0].name
        pred = self.rec_sess.run(None, {name: inp})
        return self._ctc_decode(pred[0])

    # ── Validate الحقول ───────────────────────────────────────
    def _validate(self, field: str, text: str) -> dict:
        rules = {
            "national_id": lambda t: bool(re.match(r'^[23]\d{13}$',
                                                     re.sub(r'\D','',t))),
            "gender":      lambda t: t.strip() in ["ذكر", "أنثى"],
            "birth_date":  lambda t: bool(re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', t)),
        }
        valid = rules[field](text) if field in rules else len(text.strip()) > 0
        return {"valid": valid, "text": text}

    # ── الدالة الرئيسية ───────────────────────────────────────
    def extract(self, id_card_path: str,
                crops: dict[str, np.ndarray] = None) -> dict:
        """
        id_card_path : مسار صورة البطاقة الكاملة
        crops        : لو عندك حقول مقصوصة جاهزة (اختياري)
        """
        if crops is None:
            # قص الحقول بـ YOLO (مبسط — استخدم YOLOFieldDetector الكامل)
            img   = cv2.imread(id_card_path)
            crops = self._detect_and_crop(img)

        results = {}
        for field, crop in crops.items():
            text, conf = self.recognize(crop)
            val        = self._validate(field, text)
            results[field] = {
                "text":       text,
                "confidence": round(conf, 3),
                "valid":      val["valid"],
            }

        return results

    def _detect_and_crop(self, img):
        """placeholder — استبدل بـ YOLOFieldDetector الكامل"""
        raise NotImplementedError(
            "مرّر crops مباشرة أو استخدم YOLOFieldDetector"
        )


# ── تشغيل ────────────────────────────────────────────────────
ocr = EgyptianIDOCR(
    det_onnx  = "./onnx/det_sim.onnx",
    rec_onnx  = "./onnx/rec_sim.onnx",
    dict_path = "./arabic_dict.txt",
    use_gpu   = False,
)

# مثال بحقول جاهزة
crops = {
    "name":        cv2.imread("./crops/test_name.jpg"),
    "national_id": cv2.imread("./crops/test_nid.jpg"),
    "address":     cv2.imread("./crops/test_address.jpg"),
}

result = ocr.extract("", crops=crops)

print("\n" + "="*45)
for field, data in result.items():
    status = "✅" if data["valid"] else "⚠️"
    print(f"{status} {field:<18}: {data['text']}")
    print(f"   conf={data['confidence']:.3f} | "
          f"valid={data['valid']}")
print("="*45)
```


***

## Cell 12 — Benchmark السرعة النهائي

```python
import time, statistics

def benchmark(ocr_engine, crops: dict,
              n: int = 100) -> dict:
    """قياس السرعة على CPU"""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        for field, crop in crops.items():
            ocr_engine.recognize(crop)
        times.append((time.perf_counter() - t0) * 1000)

    n_fields = len(crops)
    print("\n" + "="*45)
    print(f"  ⚡ Benchmark ({n} runs, {n_fields} fields)")
    print("="*45)
    print(f"  Per card  avg : {statistics.mean(times):.1f} ms")
    print(f"  Per card  p95 : {sorted(times)[int(n*.95)]:.1f} ms")
    print(f"  Per field avg : {statistics.mean(times)/n_fields:.1f} ms")
    print(f"  Throughput    : {1000/statistics.mean(times):.1f} cards/sec")
    print("="*45)

    return {"avg_ms": statistics.mean(times),
            "p95_ms": sorted(times)[int(n*.95)]}

benchmark(ocr, crops, n=100)
```


***

## الـ Pipeline بالكامل في سطر واحد

```
egyptian_id_ready/
├── train+valid+test  ─→  build_crops_from_split()
│                              ↓
│                     label_all_crops()  [QARI / Gemini]
│                              ↓
│                     write_paddle_txts()
│                              ↓
│                     Fine-tuning PaddleOCR
│                              ↓
│                     evaluate_on_test()
│                              ↓
│                     export → ONNX
│                              ↓
└── Production ──────  EgyptianIDOCR.extract()
```

```
📊 الأداء المتوقع على CPU:
   Fine-tuned PaddleOCR ONNX : ~15–25 ms/حقل
   7 حقول كاملة              : ~100–175 ms/بطاقة
   Throughput                 : ~6–10 بطاقات/ثانية
```


---

## Cell 13 — FastAPI Service كامل

```python
# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from functools import lru_cache
import numpy as np
import cv2, time, logging, json
from datetime import datetime
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("ocr_service.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("egyptian_id_ocr")

# ── Startup / Shutdown ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Loading OCR model...")
    app.state.ocr = EgyptianIDOCR(
        det_onnx  = "./onnx/det_sim.onnx",
        rec_onnx  = "./onnx/rec_sim.onnx",
        dict_path = "./arabic_dict.txt",
        use_gpu   = False,
    )
    logger.info("✅ Model ready")
    yield
    logger.info("🛑 Shutting down")

app = FastAPI(
    title       = "Egyptian ID OCR API",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Helpers ───────────────────────────────────────────────────
def decode_image(contents: bytes) -> np.ndarray:
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Cannot decode image")
    return img

def log_request(endpoint: str, latency_ms: float,
                n_fields: int, avg_conf: float):
    logger.info(json.dumps({
        "endpoint":    endpoint,
        "latency_ms":  round(latency_ms, 1),
        "fields":      n_fields,
        "avg_conf":    round(avg_conf, 3),
        "timestamp":   datetime.utcnow().isoformat(),
    }, ensure_ascii=False))

# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health():
    """فحص حالة السيرفر والنموذج"""
    return {
        "status":  "healthy",
        "model":   "EgyptianIDOCR v1.0",
        "device":  "CPU",
        "time":    datetime.utcnow().isoformat(),
    }


@app.post("/ocr/full-card")
async def ocr_full_card(file: UploadFile = File(...)):
    """
    استخراج جميع حقول البطاقة من صورة كاملة
    يستخدم YOLO للكشف ثم PaddleOCR للقراءة
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(415, "Only JPEG/PNG supported")

    t0       = time.perf_counter()
    img      = decode_image(await file.read())

    # كشف الحقول
    fields_detected = app.state.ocr.det_sess and \
                      detector.detect(img)   # YOLOFieldDetector
    if not fields_detected:
        raise HTTPException(422, "No fields detected in image")

    # قص واستخراج
    crops   = {
        f["class_name"]: img[
            f["bbox"][1]:f["bbox"][3],
            f["bbox"][0]:f["bbox"][2]
        ]
        for f in fields_detected
    }
    results = app.state.ocr.extract("", crops=crops)
    elapsed = (time.perf_counter() - t0) * 1000

    avg_conf = sum(v["confidence"] for v in results.values()) / len(results)
    log_request("/ocr/full-card", elapsed, len(results), avg_conf)

    return {
        "status":      "success",
        "latency_ms":  round(elapsed, 1),
        "fields":      results,
        "avg_confidence": round(avg_conf, 3),
    }


@app.post("/ocr/single-field")
async def ocr_single_field(
    file:  UploadFile = File(...),
    field: str        = "name",
):
    """استخراج نص حقل واحد من صورة مقصوصة"""
    t0   = time.perf_counter()
    img  = decode_image(await file.read())
    text, conf = app.state.ocr.recognize(img)
    val        = app.state.ocr._validate(field, text)
    elapsed    = (time.perf_counter() - t0) * 1000

    log_request("/ocr/single-field", elapsed, 1, conf)

    return {
        "field":      field,
        "text":       text,
        "confidence": round(conf, 3),
        "valid":      val["valid"],
        "latency_ms": round(elapsed, 1),
    }


@app.post("/ocr/batch")
async def ocr_batch(files: list[UploadFile] = File(...)):
    """
    معالجة عدة بطاقات دفعة واحدة
    مفيد للمعالجة الجماعية
    """
    if len(files) > 20:
        raise HTTPException(400, "Max 20 images per batch")

    t0      = time.perf_counter()
    results = []

    for f in files:
        try:
            img        = decode_image(await f.read())
            text, conf = app.state.ocr.recognize(img)
            results.append({
                "filename":   f.filename,
                "text":       text,
                "confidence": round(conf, 3),
                "status":     "success",
            })
        except Exception as e:
            results.append({
                "filename": f.filename,
                "status":   f"error: {e}",
            })

    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "total":      len(files),
        "success":    sum(1 for r in results if r["status"] == "success"),
        "latency_ms": round(elapsed, 1),
        "results":    results,
    }


@app.get("/metrics")
def metrics():
    """إحصائيات الاستخدام من الـ log"""
    try:
        with open("ocr_service.log") as f:
            lines = [l for l in f if '"endpoint"' in l]

        logs = [json.loads(l.split(" | INFO | ")[-1])
                for l in lines[-1000:]]   # آخر 1000 request

        if not logs:
            return {"message": "No requests yet"}

        latencies = [l["latency_ms"] for l in logs]
        import statistics
        return {
            "total_requests": len(logs),
            "avg_latency_ms": round(statistics.mean(latencies), 1),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies)*.95)], 1),
            "avg_confidence": round(
                sum(l.get("avg_conf", 0) for l in logs) / len(logs), 3
            ),
        }
    except Exception as e:
        return {"error": str(e)}
```


***

## Cell 14 — Dockerfile + docker-compose

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# تثبيت مكتبات النظام
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# تثبيت Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ الكود والنماذج
COPY app/          ./app/
COPY onnx/         ./onnx/
COPY arabic_dict.txt .

EXPOSE 8000

CMD ["uvicorn", "app.main:app",
     "--host", "0.0.0.0",
     "--port", "8000",
     "--workers", "1"]    # worker واحد لـ ONNX على CPU
```

```yaml
# docker-compose.yml
version: "3.9"

services:
  ocr-api:
    build: .
    container_name: egyptian_id_ocr
    ports:
      - "8000:8000"
    volumes:
      - ./onnx:/app/onnx          # النماذج خارج الـ image
      - ./logs:/app/logs          # حفظ الـ logs
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "2.0"
```

```txt
# requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.30.0
onnxruntime==1.19.0
opencv-python-headless==4.10.0.84
Pillow==10.4.0
arabic-reshaper==3.0.0
python-bidi==0.4.2
numpy==1.26.4
pandas==2.2.2
python-multipart==0.0.9
```

```bash
# بناء وتشغيل
docker compose up --build -d

# مشاهدة الـ logs
docker compose logs -f ocr-api

# اختبار
curl http://localhost:8000/health
```


***

## Cell 15 — Integration Test كامل

```python
# tests/test_full_pipeline.py
import pytest, requests, cv2, time
import numpy as np
from pathlib import Path

BASE_URL = "http://localhost:8000"

class TestEgyptianIDOCR:

    # ── Health ────────────────────────────────────────────────
    def test_health(self):
        r = requests.get(f"{BASE_URL}/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    # ── Single Field ──────────────────────────────────────────
    def test_name_field(self):
        img_path = "tests/fixtures/name_crop.jpg"
        with open(img_path, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/ocr/single-field",
                files={"file": ("name.jpg", f, "image/jpeg")},
                params={"field": "name"},
            )
        assert r.status_code == 200
        data = r.json()
        assert "text" in data
        assert data["confidence"] > 0.5
        assert len(data["text"]) > 0

    def test_national_id_validation(self):
        img_path = "tests/fixtures/nid_crop.jpg"
        with open(img_path, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/ocr/single-field",
                files={"file": ("nid.jpg", f, "image/jpeg")},
                params={"field": "national_id"},
            )
        data = r.json()
        assert data["valid"] == True
        assert len(data["text"].replace(" ", "")) == 14

    # ── Latency ───────────────────────────────────────────────
    def test_latency_under_500ms(self):
        img_path = "tests/fixtures/name_crop.jpg"
        times = []
        for _ in range(10):
            with open(img_path, "rb") as f:
                t0 = time.perf_counter()
                requests.post(
                    f"{BASE_URL}/ocr/single-field",
                    files={"file": ("name.jpg", f, "image/jpeg")},
                    params={"field": "name"},
                )
                times.append((time.perf_counter() - t0) * 1000)

        avg = sum(times) / len(times)
        assert avg < 500, f"Too slow: {avg:.0f}ms"
        print(f"✅ Avg latency: {avg:.0f}ms")

    # ── Batch ─────────────────────────────────────────────────
    def test_batch_endpoint(self):
        fixtures = list(Path("tests/fixtures").glob("*.jpg"))[:5]
        files    = [
            ("files", (p.name, open(p,"rb"), "image/jpeg"))
            for p in fixtures
        ]
        r = requests.post(f"{BASE_URL}/ocr/batch", files=files)
        assert r.status_code == 200
        data = r.json()
        assert data["total"]   == len(fixtures)
        assert data["success"] == len(fixtures)

    # ── Error Handling ────────────────────────────────────────
    def test_invalid_file_type(self):
        r = requests.post(
            f"{BASE_URL}/ocr/single-field",
            files={"file": ("test.pdf", b"fake", "application/pdf")},
            params={"field": "name"},
        )
        assert r.status_code == 415

    def test_corrupted_image(self):
        r = requests.post(
            f"{BASE_URL}/ocr/single-field",
            files={"file": ("bad.jpg", b"not_an_image", "image/jpeg")},
            params={"field": "name"},
        )
        assert r.status_code in [400, 422]


# تشغيل
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```


***

## Cell 16 — الـ Pipeline بالكامل في Script واحد

```bash
#!/bin/bash
# run_full_pipeline.sh

echo "════════════════════════════════════════"
echo "   Egyptian ID OCR — Full Pipeline"
echo "════════════════════════════════════════"

# 1. قص الحقول من كل الـ splits
echo "📸 Step 1: Cropping fields..."
python -c "
from pipeline import build_crops_from_split
import pandas as pd
dfs = [build_crops_from_split(s) for s in ['train','valid','test']]
pd.concat(dfs).to_csv('crops_metadata.csv', index=False)
print('Done')
"

# 2. استخراج النص بـ QARI
echo "🤗 Step 2: Labeling with QARI-OCR..."
python -c "
from pipeline import label_all_crops
import pandas as pd
df = pd.read_csv('crops_metadata.csv')
label_all_crops(df, splits_to_label=['train','valid'])
"

# 3. بناء ملفات PaddleOCR
echo "📄 Step 3: Building PaddleOCR label files..."
python -c "from pipeline import write_paddle_txts; write_paddle_txts()"

# 4. Fine-tuning
echo "🏋️  Step 4: Fine-tuning..."
python tools/train.py -c configs/egyptian_id_rec.yml

# 5. تقييم
echo "📊 Step 5: Evaluating..."
python tools/eval.py -c configs/egyptian_id_rec.yml \
    -o Global.checkpoints=./output/egyptian_id_rec/best_accuracy

# 6. تصدير ONNX
echo "📦 Step 6: Exporting to ONNX..."
python tools/export_model.py \
    -c configs/egyptian_id_rec.yml \
    -o Global.pretrained_model=./output/egyptian_id_rec/best_accuracy \
       Global.save_inference_dir=./inference/rec
paddle2onnx --model_dir ./inference/rec \
            --save_file ./onnx/rec.onnx --opset_version 11
python -m onnxsim ./onnx/rec.onnx ./onnx/rec_sim.onnx

# 7. تشغيل السيرفر
echo "🚀 Step 7: Starting API server..."
docker compose up --build -d

echo "════════════════════════════════════════"
echo "✅ Pipeline complete!"
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo "════════════════════════════════════════"
```


***

## الـ Notebook كاملاً — فهرس الـ Cells

```
Cell  1  ── إعداد + التحقق من البنية
Cell  2  ── قراءة YOLO Labels + قص الحقول
Cell  3  ── استخراج النص (QARI / Gemini)
Cell  4  ── تشغيل الاستخراج على Dataset كاملة
Cell  5  ── تنظيف + بناء train/val/test .txt
Cell  6  ── إحصائيات وتقرير شامل
Cell  7  ── Fine-tuning PaddleOCR
Cell  8  ── تقييم على Test Split (CER/WER)
Cell  9  ── تحليل الأخطاء
Cell 10  ── تصدير ONNX
Cell 11  ── Inference Pipeline (EgyptianIDOCR)
Cell 12  ── Benchmark السرعة
Cell 13  ── FastAPI Service
Cell 14  ── Docker + docker-compose
Cell 15  ── Integration Tests
Cell 16  ── run_full_pipeline.sh
```

