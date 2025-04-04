import requests
import glob

paths = glob.glob("./data/train/images/*", recursive=True) # returns a list of file paths
images = [open(p, 'rb') for p in paths][:1000] # or paths[:3] to select the first 3 images
url = 'http://0.0.0.0:8001/predict'
model_name = 'model_0'

for i in range(len(images)):
    files = {'image': images[i]}
    params = {}
    if model_name:
        params['model'] = model_name
    resp = requests.post(url, files=files, params=params) 
    print(resp.json())