from PIL import Image
import numpy as np
import requests
import torch
import sys
import time
import os

root_path='/dynamic_batch/triton-multi-modal-serving'

sys.path.append(root_path)
from models.blip.blip_vqa_text_decoder import blip_vqa_text_decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bs=1
if len(sys.argv) > 1:
        bs = int(sys.argv[1])
        print(f"bs={bs}")

questions_states = np.load(root_path+"/pretrained/questions_states.npy")
#print(questions_states.shape)
questions_states = np.repeat(questions_states,bs,axis=0)

questions_atts_shape=(questions_states.shape[0]*questions_states.shape[1],questions_states.shape[2])
questions_atts = torch.ones(questions_atts_shape, dtype=torch.long).numpy(force=True)
#print(questions_atts.shape)

model_url = root_path+"/pretrained/model_base_vqa_capfilt_large.pth"

model = blip_vqa_text_decoder(pretrained=model_url, vit="base")
model.eval()
#print(sum(p.numel() for p in model.parameters()))
model = model.to(device)

with torch.no_grad():
    answers = model(questions_states,questions_atts)
    #answers = model(questions_states)
#print(answers)
#start_time=time.time()
with torch.no_grad():
    answers = model(questions_states,questions_atts)
#print(answers.shape)
#end_time=time.time()
#print(f"time={end_time-start_time}")
#print(answers)
#[b'on bench']
#with open(root_path+"/blip_vqa_text_decoder_time.txt","a") as f:
 #       f.write(f"{bs},{end_time-start_time}\n")
