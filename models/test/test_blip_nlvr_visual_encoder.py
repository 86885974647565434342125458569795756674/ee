import numpy as np
import torch
import os
import sys

sys.path.append(os.path.join(os.getcwd()))
from models.blip.blip_nlvr_visual_encoder import blip_nlvr_visual_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images0=np.array([b"/workspace/demos/images/ex0_0.jpg",b"/workspace/demos/images/acorns_1.jpg"])
images1=np.array([b"/workspace/demos/images/ex0_1.jpg",b"/workspace/demos/images/acorns_6.jpg"])
model_url = "/workspace/pretrained/model_base_nlvr.pth"

model = blip_nlvr_visual_encoder(pretrained=model_url, vit="base")

model.eval()
print(sum(p.numel() for p in model.parameters()))
model = model.to(device)

with torch.no_grad():
    images_embeds = model(images0,images1)

print(images_embeds.shape)
print(images_embeds.dtype)


with open("/workspace/pretrained/blip_nlvr_images_embeds.npy", "wb") as f:
    np.save(f, images_embeds)
