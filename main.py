from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import time

app = FastAPI(
    title="YOLO11 Deployment API",
    description="A RESTful API for deploying Ultralytics YOLO11 model(s)",
    version="1.0"
)

# -------------------------
# 1. Load Your YOLO11 Model(s)
# -------------------------
# Assuming you have at least one model in the ./models directory.
# For demonstration, we load two models (they could be the same file or different).
model_0 = YOLO("./models/best.pt")
model_1 = YOLO("./models/v1.pt")  # For example, if you have a second model

models = {
    "model_0": model_0,
    "model_1": model_1
}

default_model_name = "model_0"  # Default model to use if none is specified
start_time = time.time()

# Variables for simple metrics tracking
request_count = 0
total_latency = 0.0
max_latency = 0.0
request_timestamps = []  # list to track request times for request rate calculation

# -------------------------
# 2. Helper Functions
# -------------------------
def get_uptime():
    elapsed = time.time() - start_time
    days = int(elapsed // 86400)
    hours = int((elapsed % 86400) // 3600)
    minutes = int((elapsed % 3600) // 60)
    return f"{days} days, {hours} hours, {minutes} minutes"

def process_image(image_bytes: bytes) -> np.ndarray:
    """Converts raw image bytes into a NumPy image (BGR) using OpenCV."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

# -------------------------
# 3. API Endpoints
# -------------------------

# Health Check Endpoint
@app.get("/health-status")
def health_status():
    return {
        "status": "Healthy",
        "server": "FastAPI",
        "uptime": get_uptime()
    }

# Model Management Endpoint (List Models)
@app.get("/management/models")
def list_models():
    return {
        "available_models": list(models.keys())
    }

# Group Info Endpoint
@app.get("/group-info")
def group_info():
    return {
        "group": "Group 2",
        "members": ["Fuad Hasan", "Junfeng Lei", "Zhibo Wang", "Tony Zhao"]
    }

# Model Info Endpoint (Describe a Model)
@app.get("/management/models/{model}/describe")
def describe_model(model: str):
    dummy_configs = {
        "model_0": {
            "input_size": [640, 640],
            "batch_size": 16,
            "confidence_threshold": 0.25
        },
        "model_1": {
            "input_size": [640, 640],
            "batch_size": 16,
            "confidence_threshold": 0.30
        }
    }
    if model not in models:
        return JSONResponse(status_code=404, content={"error": "Model not found."})
    return {
        "model": model,
        "config": dummy_configs.get(model, {}),
        "date_registered": "2025-03-26"  # This can be dynamically set if desired
    }

# Change Default Model Endpoint
@app.get("/management/models/{model}/set-default")
def set_default_model(model: str):
    global default_model_name
    if model not in models:
        return JSONResponse(status_code=404, content={"error": "Model not found."})
    default_model_name = model
    return {
        "success": True,
        "default_model": default_model_name
    }

# Inference Endpoint (/predict)
@app.post("/predict")
def predict(image: UploadFile = File(...), model: str = Query(None, description="YOLO model name to use")):
    global request_count, total_latency, max_latency

    start_request_time = time.time()
    request_timestamps.append(start_request_time)
    
    # Choose model
    if model and model in models:
        chosen_model = models[model]
        chosen_model_name = model
    else:
        chosen_model = models[default_model_name]
        chosen_model_name = default_model_name

    # Read image file from request
    contents = image.file.read()
    img = process_image(contents)
    
    # Run inference using Ultralytics YOLO11 predict mode
    # Here, stream=False returns a list of Results objects
    results = chosen_model.predict(img, conf=0.25, imgsz=640)
    
    predictions = []
    # Loop through results (for each image; typically one image per request)
    for r in results:
        # Each result contains a Boxes object
        for box in r.boxes:
            # Extract box coordinates in xyxy format, confidence, and class index
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = chosen_model.names[cls_id]
            predictions.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2]
            })

    elapsed = time.time() - start_request_time
    request_count += 1
    total_latency += elapsed
    if elapsed > max_latency:
        max_latency = elapsed

    return {
        "predictions": predictions,
        "model_used": chosen_model_name
    }

# Metrics Endpoint
@app.get("/metrics")
def get_metrics():
    global request_count, total_latency, max_latency
    now = time.time()
    one_min_ago = now - 60
    while request_timestamps and request_timestamps[0] < one_min_ago:
        request_timestamps.pop(0)
    request_rate = len(request_timestamps)
    avg_latency = (total_latency / request_count * 1000) if request_count else 0.0
    max_latency_ms = max_latency * 1000
    return {
        "request_rate_per_minute": request_rate,
        "avg_latency_ms": round(avg_latency, 2),
        "max_latency_ms": round(max_latency_ms, 2),
        "total_requests": request_count
    }

# -------------------------
# 4. Run the Application
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
