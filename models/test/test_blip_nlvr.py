from PIL import Image
import numpy as np
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
import sys

sys.path.append(os.path.join(os.getcwd()))
from models.blip.blip_nlvr import blip_nlvr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images0=np.array([b"/workspace/demos/images/ex0_0.jpg",b"/workspace/demos/images/acorns_1.jpg"])
images1=np.array([b"/workspace/demos/images/ex0_1.jpg",b"/workspace/demos/images/acorns_6.jpg"])


questions = np.array(
    [b"The left image contains twice the number of dogs as the right image, and at least two dogs in total are standing.",
    b"One image shows exactly two brown acorns in back-to-back caps on green foliage.",]
)


model_url = "/workspace/pretrained/model_base_nlvr.pth"

model = blip_nlvr(pretrained=model_url, vit="base")

model.eval()
print(sum(p.numel() for p in model.parameters()))
model = model.to(device)

with torch.no_grad():
    predictions = model(images0, images1, questions)
print(predictions)
