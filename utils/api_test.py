import os
import requests
import time
import random


def send_image(image_path):
    url = "http://127.0.0.1:8000/predict"
    headers = {"accept": "application/json"}
    try:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(url, files=files, headers=headers)
        if response.status_code == 200:
            print(f"Successfully sent {image_path}")
        else:
            print(f"Failed to send {image_path}: {response.status_code}")
    except Exception as e:
        print(f"Error sending {image_path}: {e}")


def main():
    folder_path = "./output/ref_best/images"
    iteration = 100  # Number of iterations to send images

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a directory")
        return

    image_files = [f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg"))]

    if not image_files:
        print("No images found in the directory.")
        return

    for _ in range(iteration):
        # Randomly select an image from the list
        random_file_name = random.choice(image_files)

        image_path = os.path.join(folder_path, random_file_name)
        send_image(image_path)
        time.sleep(1)


if __name__ == "__main__":
    main()
