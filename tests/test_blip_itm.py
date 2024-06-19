import os
import sys

sys.path.append(os.path.join(os.getcwd()))

import numpy as np
import torch

from models import blip_itm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
model_url = "/workspace/pretrained/model_base_retrieval_coco.pth"

image_urls = np.array([
    b"/workspace/demos/images/beach.jpg",
    b"/workspace/demos/images/beach.jpg",
])

captions = np.array([
    b'a woman sitting on the beach with a dog',
    b'a woman sitting on the beach with a dog',
])
model = blip_itm(pretrained=model_url)
model.eval()
model = model.to(device)

with torch.no_grad():
    output = model(image_urls, captions)
print(output)
