
# YOLO11 Deployment API

This repository contains a FastAPI application that deploys two trained Ultralytics YOLO11 models as a RESTful API. 

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the API](#running-the-api)
  - [Available Endpoints](#available-endpoints)
- [API Endpoints Details](#api-endpoints-details)
- [Next Steps for the Assignment](#next-steps-for-the-assignment)
- [Monitoring & Dashboard Development](#monitoring--dashboard-development)
- [Report and Deliverables](#report-and-deliverables)
- [License](#license)

## Overview

This project demonstrates how to deploy multiple YOLO11 models using FastAPI. The RESTful API allows clients to send images (JPEG, PNG, etc.) and receive object detection predictions:
- Packaging at least two trained models.
- Providing endpoints for inference, health checks, model management, group information, and performance metrics.
- Implementing basic performance monitoring.

## Features

- **Multiple Model Deployment:** Two models (`best.pt` and `v1.pt`) are loaded and managed via the API.
- **Flexible Inference:** Accepts various image formats and sizes. Images are preprocessed with OpenCV.
- **Standardized JSON Responses:** Prediction results (bounding boxes, confidence scores, and labels) are returned in a specified JSON format.
- **Endpoints:** Includes health check, model list, group info, model details, default model management, inference, and metrics.
- **Basic Monitoring:** Tracks API request rate, average latency, maximum latency, and total requests.

## Installation

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda (recommended for environment management)
- Python 3.9 or later

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/fhasanm/yolo_api.git
   cd yolo_api
   ```

2. **Create and activate a new Conda environment:**

   ```bash
   conda create -n yolo_api_env python=3.9 -y
   conda activate yolo_api_env
   ```

3. **Install dependencies:**

   Ensure your `requirements.txt` contains:
   ```txt
   fastapi
   uvicorn
   pillow
   opencv-python
   numpy
   torch
   ultralytics
   python-multipart
   ```
   
   Then install:
   ```bash
   pip install -r requirements.txt
   ```

4. **Place your trained model files** in the `./models` directory (e.g., `best.pt` and `v1.pt`).

## Usage

### Running the API

Start the FastAPI application locally using Uvicorn:

```bash
uvicorn main:app --reload
```

The API will be running at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### Available Endpoints

- **Health Check:** `/health-status` (GET)  
  Returns the server status and uptime.
- **List Models:** `/management/models` (GET)  
  Returns a list of available models.
- **Group Info:** `/group-info` (GET)  
  Returns group information.
- **Model Details:** `/management/models/{model}/describe` (GET)  
  Provides details about a specific model.
- **Change Default Model:** `/management/models/{model}/set-default` (GET)  
  Sets the specified model as the default.
- **Inference:** `/predict` (POST)  
  Accepts an image file (and optional model name) and returns detection predictions.
- **Metrics:** `/metrics` (GET)  
  Returns performance metrics such as request rate and latency.

You can test these endpoints interactively by navigating to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), which is the automatically generated Swagger UI.

## API Endpoints Details

### `/predict` Endpoint

- **Method:** POST  
- **Request Body:**  
  - `image`: file  
  - `model`: text (optional; if not provided, default model is used)
- **Response Example:**

  ```json
  {
    "predictions": [
      {
        "label": "object_class",
        "confidence": 0.91,
        "bbox": [42, 58, 172, 310]
      }
    ],
    "model_used": "model_0"
  }
  ```

### `/health-status` Endpoint

- **Method:** GET  
- **Response Example:**

  ```json
  {
    "status": "Healthy",
    "server": "FastAPI",
    "uptime": "0 days, 0 hours, 5 minutes"
  }
  ```

### `/metrics` Endpoint

- **Method:** GET  
- **Response Example:**

  ```json
  {
    "request_rate_per_minute": 42,
    "avg_latency_ms": 153.4,
    "max_latency_ms": 312.7,
    "total_requests": 1200
  }
  ```

(Other endpoints follow similar JSON formats as per the assignment requirements.)

## Next Steps for the Assignment

1. **Enhanced Monitoring & Dashboard Development:**
   - **Real-Time Metrics:** Integrate Prometheus and Grafana to continuously monitor API response times, request rates, and model performance.
   - **Concept Drift Monitoring:** Optionally, integrate tools like Evidently AI to track changes in model prediction confidence over time and detect concept drift.
   - **Custom Dashboards:** Develop custom dashboards to visualize model confidence distributions and inference accuracy, potentially using tools such as Apache Airflow for scheduled drift evaluations.

2. **Evaluation Script Integration:**
   - Ensure that our API endpoints are compliant with the provided evaluation script. Run the evaluation script on the same machine as our API and include the generated report in our submission.


