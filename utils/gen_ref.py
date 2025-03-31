import pandas as pd
from ultralytics import YOLO
import cv2
import os
import time

# Load the model (e.g., model_0)
model = YOLO("./models/best.pt")
model_name = "model_0"

# List of reference image paths
reference_folder = "data/ref_best/images"

predictions = []

# iterate all images under reference_folder
for filename in os.listdir(reference_folder):
    # not png or jpg
    if not filename.endswith(('.png', '.jpg', '.jpeg')):
        continue

    # get full path
    image_path = os.path.join(reference_folder, filename)

    img = cv2.imread(image_path)
    results = model.predict(img, conf=0.25, imgsz=640)
    timestamp = time.time()  # Use a fixed or current timestamp
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            predictions.append({"timestamp": timestamp, "model_name": model_name, "class": label, "confidence": round(conf, 2)})

# Save to reference_predictions.csv
df = pd.DataFrame(predictions)
df.to_csv("reference_predictions.csv", index=False)
