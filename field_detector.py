"""
YOLO ONNX Field Detector
=========================
Detect ID card fields using field_detector.onnx when no labels exist.
"""

import numpy as np
import onnxruntime as ort

from label_reader import get_field_name


class YOLOFieldDetector:
    """
    Run YOLO ONNX inference to detect ID card fields.
    Used only for images that lack pre-existing YOLO labels.
    """

    def __init__(
        self,
        onnx_path: str,
        input_size: int = 640,
        conf_thresh: float = 0.35,
        field_mapping: dict = None,
    ):
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.field_mapping = field_mapping
        self.input_name = self.session.get_inputs()[0].name
        print(
            f"✅ YOLO loaded | Provider: {self.session.get_providers()[0]}"
        )

    # ── Letterbox preprocessing ───────────────────────────────
    def _preprocess(self, img: np.ndarray):
        h, w = img.shape[:2]
        import cv2

        scale = self.input_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        canvas = np.full(
            (self.input_size, self.input_size, 3), 114, dtype=np.uint8
        )
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        blob = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[np.newaxis], scale, pad_x, pad_y

    # ── NMS ───────────────────────────────────────────────────
    @staticmethod
    def _nms(boxes, scores, iou_threshold=0.4):
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= iou_threshold]

        return keep

    # ── Main detection ────────────────────────────────────────
    def detect(self, img: np.ndarray) -> list[dict]:
        """
        Returns list of dicts:
            class_id, class_name, bbox [x1,y1,x2,y2], conf, source
        """
        h_orig, w_orig = img.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(img)

        outputs = self.session.run(None, {self.input_name: blob})[0]
        predictions = outputs[0]

        boxes_out, scores_out, class_ids_out = [], [], []

        for pred in predictions:
            obj_conf = pred[4]
            if obj_conf < self.conf_thresh:
                continue

            class_scores = pred[5:]
            class_id = int(np.argmax(class_scores))
            confidence = float(obj_conf * class_scores[class_id])
            if confidence < self.conf_thresh:
                continue

            cx, cy, bw, bh = pred[:4]
            x1 = int((cx - bw / 2 - pad_x) / scale)
            y1 = int((cy - bh / 2 - pad_y) / scale)
            x2 = int((cx + bw / 2 - pad_x) / scale)
            y2 = int((cy + bh / 2 - pad_y) / scale)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)

            boxes_out.append([x1, y1, x2, y2])
            scores_out.append(confidence)
            class_ids_out.append(class_id)

        if not boxes_out:
            return []

        keep = self._nms(np.array(boxes_out), np.array(scores_out))
        results = []
        for i in keep:
            cid = class_ids_out[i]
            results.append({
                "class_id": cid,
                "class_name": get_field_name(cid, self.field_mapping),
                "bbox": boxes_out[i],
                "conf": round(scores_out[i], 3),
                "source": "yolo",
            })
        return results
