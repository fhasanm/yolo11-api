import pandas as pd
from ultralytics import YOLO
import cv2
import os
import time
import uuid

# Load the model (e.g., model_0)
model = YOLO("./models/best.pt")
model_name = "model_0"

# List of reference image paths
reference_folder = "data/ref_best/images"

csv_content = []

timestamp = time.time()  # Use a fixed timestamp

# iterate all images under reference_folder
for filename in os.listdir(reference_folder):
    # not png or jpg
    if not filename.endswith(('.png', '.jpg', '.jpeg')):
        continue

    # get full path
    image_path = os.path.join(reference_folder, filename)
    ref_uuid = str(uuid.uuid4())  # Generate a unique UUID for each image

    img = cv2.imread(image_path)
    results = model.predict(img, conf=0.25, imgsz=640)
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)

            csv_content.append(
                {
                    "request_id": ref_uuid,
                    "timestamp": timestamp,
                    "model_name": model_name,
                    "class": label,
                    "confidence": conf,
                    "area": area,
                }
            )

# Save to reference_predictions.csv
df = pd.DataFrame(csv_content)
df.to_csv("ref_pred_0.csv", index=False)
