"""
Construction Detection API — Hugging Face Space
Loads model from HF Hub, serves REST API for mobile app
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import numpy as np
import cv2, base64, time, os

# ── CONFIG ──────────────────────────────────────────────────────────────────
HF_REPO_ID  = "dipangshuborah/construction-detection-yolov8"
MODEL_FILE  = "best_v2_finetune.pt"
CONF        = 0.25
IOU         = 0.45

COLORS = {
    "beam":   [255, 0,   0  ],
    "column": [0,   255, 255],
    "door":   [255, 0,   255],
    "floor":  [0,   165, 255],
    "stairs": [0,   255, 0  ],
    "wall":   [255, 255, 0  ],
    "window": [0,   0,   255],
}

# ── APP ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Construction Detection API",
    description = "Detects construction elements and measures dimensions",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── GLOBAL STATE ─────────────────────────────────────────────────────────────
model         = None
pixels_per_cm = None

# ── STARTUP ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def load_model():
    global model
    print(f"Downloading {MODEL_FILE} from {HF_REPO_ID}...")
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
    model = YOLO(path)
    print("✅ Model loaded!")

# ── HELPERS ──────────────────────────────────────────────────────────────────
def bytes_to_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def image_to_base64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")

def px_to_cm(pixels: float) -> float | None:
    if pixels_per_cm is None:
        return None
    return round(pixels / pixels_per_cm, 1)

def draw_boxes(img: np.ndarray, detections: list) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls   = det["class"]
        conf  = det["confidence"]
        color = COLORS.get(cls, [255, 255, 255])

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if det.get("width_cm"):
            label = f"{cls} {conf:.2f} W:{det['width_cm']}cm H:{det['height_cm']}cm"
        else:
            label = f"{cls} {conf:.2f} W:{det['width_px']}px H:{det['height_px']}px"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img

def run_detection(img: np.ndarray) -> list:
    results    = model.predict(img, conf=CONF, iou=IOU, task="detect", verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls  = model.names[int(box.cls)]
            conf = round(float(box.conf), 3)
            w_px = x2 - x1
            h_px = y2 - y1
            detections.append({
                "class":      cls,
                "confidence": conf,
                "bbox":       [x1, y1, x2, y2],
                "width_px":   w_px,
                "height_px":  h_px,
                "width_cm":   px_to_cm(w_px),
                "height_cm":  px_to_cm(h_px),
                "color":      COLORS.get(cls, [255, 255, 255]),
            })
    return detections

# ── ROUTES ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status":    "running",
        "model":     MODEL_FILE,
        "classes":   list(COLORS.keys()),
        "endpoints": {
            "POST /detect":    "Upload image → detections + dimensions",
            "POST /calibrate": "Set reference object for real-world units",
            "GET  /health":    "Health check",
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/calibrate")
async def calibrate(
    file:        UploadFile = File(...),
    bbox_x1:     int   = 0,
    bbox_y1:     int   = 0,
    bbox_x2:     int   = 210,
    bbox_y2:     int   = 297,
    real_width:  float = 21.0,
    real_height: float = 29.7,
):
    """
    Calibrate using a reference object (e.g. A4 paper = 21cm x 29.7cm).
    Provide bounding box of the reference object in pixels.
    """
    global pixels_per_cm
    data = await file.read()
    img  = bytes_to_image(data)
    if img is None:
        raise HTTPException(400, "Invalid image")

    ref_px_w  = bbox_x2 - bbox_x1
    ref_px_h  = bbox_y2 - bbox_y1
    px_per_w  = ref_px_w / real_width
    px_per_h  = ref_px_h / real_height
    pixels_per_cm = round((px_per_w + px_per_h) / 2, 4)

    return {
        "message":       "✅ Calibration successful",
        "pixels_per_cm": pixels_per_cm,
    }

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Upload a construction site image.
    Returns all detected objects with bounding boxes and dimensions.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded")

    data = await file.read()
    img  = bytes_to_image(data)
    if img is None:
        raise HTTPException(400, "Invalid image")

    start      = time.time()
    detections = run_detection(img)
    elapsed    = round(time.time() - start, 3)

    annotated  = draw_boxes(img.copy(), detections)
    img_b64    = image_to_base64(annotated)

    return JSONResponse({
        "success":          True,
        "total":            len(detections),
        "inference_time_s": elapsed,
        "calibrated":       pixels_per_cm is not None,
        "image_base64":     img_b64,
        "detections":       detections,
    })
