import json
import numpy as np
import torch
import os
import sys
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import time

root_path='/dynamic_batch/ee/'
dataset_dir = root_path+"datasets/vqa/"

json_file = dataset_dir+"vqa_test.json"

with open(json_file) as f:
    dataset = json.load(f)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),                            
        (0.26862954, 0.26130258, 0.27577711),
    ),
])

#image_urls=[bytes(os.path.join(dataset_dir, dataset[i]["image"].replace("test2015/", "test2015/re_")), "utf-8") for i in range(100)]
image_urls=[bytes(os.path.join(dataset_dir, dataset[i]["image"].replace("test2015/", "test2015/re_")), "utf-8") for i in range(1000)]
#difference is big, time.perf err or os scheule err

image_urls=list(set(image_urls))

print(len(image_urls))

size=[0]*len(image_urls)
open_time=[0]*len(image_urls)
pre_time=[0]*len(image_urls)
for i,image_url in enumerate(image_urls):
    # print(image_url.decode())
    # print(f"{os.path.getsize(image_url.decode())//1024}KB")
    size[i]=os.path.getsize(image_url.decode())//1024

    start=time.perf_counter()
    image=Image.open(image_url.decode()).convert("RGB") 
    # print(f"open:{time.perf_counter()-start}")
    open_time[i]=time.perf_counter()-start

    start=time.perf_counter()
    transform(image)
    # print(f"image preprocess:{time.perf_counter()-start}")
    pre_time[i]=time.perf_counter()-start

# print(size)
# print(open_time)
# print(pre_time)

combined_lists = list(zip(size, open_time, pre_time))
sorted_lists = sorted(combined_lists, key=lambda x: x[0])

for item in sorted_lists:
    print(f"{item[0]:<4} {item[1]:<30} {item[2]:<20}")
