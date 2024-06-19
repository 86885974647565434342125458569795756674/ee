import os
import sys


import numpy as np
import torch

root_path='/dynamic_batch/triton-multi-modal-serving'

sys.path.append(root_path)
from models.blip.blip_vqa import blip_vqa


image_urls = np.array([
    root_path.encode('utf-8')+b"/demos/images/beach.jpg",
    root_path.encode('utf-8')+b"/demos/images/beach.jpg",
    root_path.encode('utf-8')+b"/demos/images/merlion.png",
    root_path.encode('utf-8')+b"/demos/images/merlion.png",
])
questions = np.array([
    b"where is the woman sitting?",
    b"where is the dog sitting?",
    b"",
    b"which city is this photo taken?",
])

model_url = root_path+"/pretrained/model_base_vqa_capfilt_large.pth"
# model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = blip_vqa(pretrained=model_url)
model.eval()
model = model.to(device)

with torch.no_grad():
    answers = model(image_urls, questions)
print(image_urls)
print(questions)
print(answers)
