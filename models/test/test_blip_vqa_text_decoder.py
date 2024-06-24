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

questions_states=torch.load(root_path+"pretrained/questions_states.pth")
questions_states=questions_states.repeat(questions_states.shape[0]*64,*tuple([1]*len(questions_states.shape[1:])))

questions_atts_shape=(questions_states.shape[0]*questions_states.shape[1],questions_states.shape[2])
questions_atts = torch.ones(questions_atts_shape, dtype=torch.long)
#print(questions_atts.shape)

model_url = root_path+"pretrained/model_base_vqa_capfilt_large.pth"

model = blip_vqa_text_decoder(pretrained=model_url, vit="base")
model.eval()
#print(sum(p.numel() for p in model.parameters()))

model.print_time=False
with torch.no_grad():
    answers = model(questions_states,questions_atts)
    #answers = model(questions_states)
#print(answers)

questions_states=torch.load(root_path+"pretrained/questions_states.pth")
questions_states=questions_states.repeat(questions_states.shape[0]*bs,*tuple([1]*len(questions_states.shape[1:])))

questions_atts_shape=(questions_states.shape[0]*questions_states.shape[1],questions_states.shape[2])
questions_atts = torch.ones(questions_atts_shape, dtype=torch.long)

#model.print_time=True
model.print_time=False
with torch.no_grad():
    if not model.print_time:
        '''
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        start.record()
        '''
        start= time.perf_counter()
    answers = model(questions_states,questions_atts)
    if not model.print_time:
        '''
        end.record()
        torch.cuda.synchronize()
        print(f"time={start.elapsed_time(end)/1000}")
        '''
        torch.cuda.synchronize()
        print(f"time={ time.perf_counter()-start}")
#print(answers)
#[b'on bench']
#with open(root_path+"/blip_vqa_text_decoder_time.txt","a") as f:
 #       f.write(f"{bs},{end_time-start_time}\n")
