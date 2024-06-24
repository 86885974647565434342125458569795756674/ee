import json
import numpy as np
import torch
import os
import sys
import time

root_path='/dynamic_batch/ee/'
dataset_dir = root_path+"datasets/vqa/"

sys.path.append(root_path)
from models.blip.blip_vqa_visual_encoder import blip_vqa_visual_encoder

json_file = dataset_dir+"vqa_test.json"

with open(json_file) as f:
    dataset = json.load(f)


bs=1
if len(sys.argv) > 1:
    bs = int(sys.argv[1])
print(f"bs={bs}")

model_url = root_path+"pretrained/model_base_vqa_capfilt_large.pth"
model = blip_vqa_visual_encoder(pretrained=model_url, vit="base")
model.eval()

with torch.no_grad():
    images_embeds = model([bytes(os.path.join(dataset_dir, dataset[0]["image"]), "utf-8")]*64)
torch.cuda.synchronize()

images=[bytes(os.path.join(dataset_dir, dataset[0]["image"]), "utf-8")]*bs

with torch.no_grad():
    images_embeds = model.forward_time(images)

with torch.no_grad():
    start=time.perf_counter()
    images_embeds = model(images)
    torch.cuda.synchronize()
    print(time.perf_counter()-start)

#torch.save(images_embeds[:1],root_path+'pretrained/images_embeds.pth')
