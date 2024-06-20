import matplotlib.pyplot as plt
import numpy as np
import sys

root_path="/dynamic_batch/triton-multi-modal-serving"

def prepare(file_name):
    x_data=[]
    y_data=[]
    with open(root_path+'/'+file_name+'.txt', 'r') as f:
        for line in f:
            x, y = line.strip().split(',')
            x_data.append(int(x))
            y_data.append(float(y))
    return x_data,y_data

x_min,x_max=100,0

plt.figure()
for sys_argv in ['blip_vqa_visual_encoder','blip_vqa_text_encoder','blip_vqa_text_decoder']:
    file_name=sys_argv+"_time"
    x_data,y_data=prepare(file_name)
    plt.plot(x_data, y_data, label=file_name)
    x_min=min(min(x_data),x_min)
    x_max=max(max(x_data),x_max)

plt.xlabel('bs')
plt.ylabel('time')
plt.title("blip_vqa_time")
plt.grid(True)
plt.legend()
plt.xticks(np.arange(x_min, x_max + 1, 1))
plt.savefig(root_path+"/"+"blip_vqa_time"+'.png') 

plt.figure()
for sys_argv in ['blip_vqa_visual_encoder','blip_vqa_text_encoder','blip_vqa_text_decoder']:
    file_name=sys_argv+"_time"
    x_data,y_data=prepare(file_name)
    y_data=[x_data[i]/y_data[i] for i in range(len(x_data))]
    plt.plot(x_data, y_data, label=file_name)
    x_min=min(min(x_data),x_min)
    x_max=max(max(x_data),x_max)

plt.xlabel('bs')
plt.ylabel('req/second')
plt.title("blip_vqa_throughput")
plt.grid(True)
plt.legend()
plt.xticks(np.arange(x_min, x_max + 1, 1))
plt.savefig(root_path+"/"+"blip_vqa_throughput"+'.png') 
