import numpy as np
import torch
import os
import sys
import time

root_path='/dynamic_batch/ee/'

sys.path.append(root_path)
from models.blip.blip_vqa_visual_encoder import blip_vqa_visual_encoder

#image_size = 480
#image2 = load_example_image(image_size=image_size)
#print(image2.shape,image2.dtype)
#images=torch.cat([image2,image2]).reshape(2,*image2.shape).numpy()
#print(images.shape, images.dtype)
#torch.Size([3, 480, 480]) torch.float32
#(2, 3, 480, 480) float32


bs=1
if len(sys.argv) > 1:
    bs = int(sys.argv[1])

print(f"bs={bs}")

images=[root_path.encode('utf-8')+b"/demos/images/merlion.png"]*bs

model_url = root_path+"pretrained/model_base_vqa_capfilt_large.pth"
model = blip_vqa_visual_encoder(pretrained=model_url, vit="base")

model.eval()
#print(sum(p.numel() for p in model.parameters()))

model.print_time=False
with torch.no_grad():
    images_embeds = model([root_path.encode('utf-8')+b"/demos/images/merlion.png"]*64)

model.print_time=True
#model.print_time=False
with torch.no_grad():
    if not model.print_time:
        ''' 
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        start.record()
        '''
        start=time.perf_counter()
    images_embeds = model(images)
    if not model.print_time:
        '''
        end.record()
        torch.cuda.synchronize()
        print(f"time={start.elapsed_time(end)/1000}")
        '''
        torch.cuda.synchronize()
        print(time.perf_counter()-start)
#(2, 901, 768) float32

#with open(root_path+"pretrained/images_embeds.npy", "wb") as f:
 #   np.save(f, images_embeds)

#torch.save(images_embeds[:1],root_path+ 'pretrained/images_embeds.pth')
