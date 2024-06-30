import numpy as np
import requests
import torch
import sys
import time
import os

root_path='/dynamic_batch/ee/'

sys.path.append(root_path)
from models.blip.blip_vqa_text_decoder import blip_vqa_text_decoder

bs=1
if len(sys.argv) > 1:
    bs = int(sys.argv[1])
print(f"bs={bs}")

#questions_states = np.load(root_path+"pretrained/questions_states.npy")
#print(questions_states.shape)
#questions_states = np.repeat(questions_states,bs,axis=0)

model_url = root_path+"pretrained/model_base_vqa_capfilt_large.pth"
model = blip_vqa_text_decoder(pretrained=model_url, vit="large")
model.eval()

questions_states=torch.load(root_path+"pretrained/questions_states.pth")
questions_states=questions_states.repeat(questions_states.shape[0]*64,*tuple([1]*len(questions_states.shape[1:])))

with torch.no_grad():
    answers = model(questions_states)
torch.cuda.synchronize()

#print(sum(p.numel() for p in model.parameters()))

questions_states=torch.load(root_path+"pretrained/questions_states.pth")
questions_states=questions_states.repeat(questions_states.shape[0]*bs,*tuple([1]*len(questions_states.shape[1:])))

with torch.no_grad():
    answers = model.forward_time(questions_states)
    #answers = model(questions_states)
#print(answers)

with torch.no_grad():
    start= time.perf_counter()
    answers = model(questions_states)
    torch.cuda.synchronize()
    print(f"time={ time.perf_counter()-start}")
print(answers)
