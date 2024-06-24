import numpy as np
import requests
import torch
import sys
import time
import os

root_path='/dynamic_batch/ee/'

sys.path.append(root_path)
from models.blip.blip_vqa_text_encoder import blip_vqa_text_encoder

bs=1
if len(sys.argv) > 1:
    bs = int(sys.argv[1])
print(f"bs={bs}")

model_url = root_path+"pretrained/model_base_vqa_capfilt_large.pth"
model = blip_vqa_text_encoder(pretrained=model_url, vit="base")
model.eval()

images_embeds = torch.load(root_path+"pretrained/images_embeds.pth")
images_embeds=images_embeds.repeat(images_embeds.shape[0]*64,*tuple([1]*len(images_embeds.shape[1:])))
questions = [b"where is the woman sitting?"]*64
with torch.no_grad():
     questions_states = model(images_embeds, questions)
torch.cuda.synchronize()
#print(questions_states.shape,questions_states.dtype)
#(2, 1, 9, 768) float32
#with open(root_path+"/pretrained/questions_states.npy", "wb") as f:
 #    np.save(f, questions_states)

images_embeds = torch.load(root_path+"pretrained/images_embeds.pth")
images_embeds=images_embeds.repeat(images_embeds.shape[0]*bs,*tuple([1]*len(images_embeds.shape[1:])))
questions = [b"where is the woman sitting?"]*bs
with torch.no_grad():
    questions_states = model.forward_time(images_embeds, questions)

with torch.no_grad():
    start= time.perf_counter()
    questions_states = model(images_embeds, questions)
    torch.cuda.synchronize()
    print(f"time={ time.perf_counter()-start}")

#with open(root_path+"/blip_vqa_text_encoder_time.txt","a") as f:
 #       f.write(f"{bs},{end_time-start_time}\n")
#torch.save(questions_states[:1],root_path+'pretrained/questions_states.pth')
