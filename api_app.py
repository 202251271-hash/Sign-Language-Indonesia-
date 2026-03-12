import base64
import os
import pickle
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ultralytics import YOLO


YOLO_MODEL_PATH = r"runs/detect/train/weights/best.pt"
RF_MODEL_PATH = r"artifacts/rf_model.pkl"
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
CONF_THRES = 0.55

CLASS_NAME_FIX = {"objects": "O"}


class PredictRequest(BaseModel):
    image_base64: str
    max_hands: int = 2


class HandPrediction(BaseModel):
    label: str
    yolo_conf: float
    rf_conf: Optional[float]
    bbox: List[int]


class PredictResponse(BaseModel):
    phrase: str
    predictions: List[HandPrediction]


app = FastAPI(title="SIBI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _decode_base64_image(image_base64: str) -> np.ndarray:
    payload = image_base64
    if "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Base64 tidak valid: {exc}")

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Gagal decode gambar.")
    return frame


def _clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def _expand_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int, margin=0.25):
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * margin)
    my = int(bh * margin)
    return _clamp_bbox(x1 - mx, y1 - my, x2 + mx, y2 + my, w, h)


def _extract_feature_63d(roi_bgr: np.ndarray, hands) -> Optional[np.ndarray]:
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0].landmark
    vec = []
    for p in lm:
        vec.extend([float(p.x), float(p.y), float(p.z)])
    if len(vec) != 63:
        return None
    return np.asarray(vec, dtype=np.float32)


if not os.path.exists(YOLO_MODEL_PATH):
    raise RuntimeError(f"YOLO model tidak ditemukan: {YOLO_MODEL_PATH}")
if not os.path.exists(RF_MODEL_PATH):
    raise RuntimeError(f"RF model tidak ditemukan: {RF_MODEL_PATH}")

yolo_model = YOLO(YOLO_MODEL_PATH)
with open(RF_MODEL_PATH, "rb") as f:
    rf_payload = pickle.load(f)
rf_model = rf_payload["model"] if isinstance(rf_payload, dict) else rf_payload

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.35,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def web_index():
    index_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="web/index.html tidak ditemukan.")
    return FileResponse(index_path)


@app.get("/app.js")
def web_app_js():
    js_path = os.path.join(WEB_DIR, "app.js")
    if not os.path.exists(js_path):
        raise HTTPException(status_code=404, detail="web/app.js tidak ditemukan.")
    return FileResponse(js_path, media_type="application/javascript")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    frame = _decode_base64_image(req.image_base64)
    h, w = frame.shape[:2]

    results = yolo_model.predict(frame, conf=CONF_THRES, imgsz=640, verbose=False)
    r = results[0]

    detections = []
    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            yolo_conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            x1, y1, x2, y2 = _expand_bbox(x1, y1, x2, y2, w, h, margin=0.25)
            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]
            feat = _extract_feature_63d(roi, hands)
            if feat is None:
                continue

            x = feat.reshape(1, -1)
            pred_label = str(rf_model.predict(x)[0])
            pred_label = CLASS_NAME_FIX.get(pred_label, pred_label)

            rf_conf = None
            if hasattr(rf_model, "predict_proba"):
                probs = rf_model.predict_proba(x)[0]
                rf_conf = float(np.max(probs))

            detections.append(
                {
                    "label": pred_label,
                    "yolo_conf": yolo_conf,
                    "rf_conf": rf_conf,
                    "bbox": [x1, y1, x2, y2],
                }
            )

    detections = sorted(detections, key=lambda d: d["yolo_conf"], reverse=True)[: max(1, req.max_hands)]
    detections = sorted(detections, key=lambda d: d["bbox"][0])
    phrase = " ".join(d["label"] for d in detections)

    return PredictResponse(phrase=phrase, predictions=detections)
