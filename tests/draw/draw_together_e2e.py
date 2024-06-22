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
y_sum=None
plt.figure()
for sys_argv in ['blip_vqa_visual_encoder_e2e','blip_vqa_text_encoder_e2e','blip_vqa_text_decoder_e2e']:
    file_name=sys_argv+"_time"
    x_data,y_data=prepare(file_name)
    if y_sum is None:
        y_sum=y_data
    else:
        y_sum=[y_sum[i]+y_data[i] for i in range(len(y_data))]
    x_min=min(min(x_data),x_min)
    x_max=max(max(x_data),x_max)

plt.plot(x_data, y_sum, label="3_sum")

x_data,y_data=prepare("blip_vqa_bls_e2e_time")
plt.plot(x_data, y_data, label="bls")

plt.xlabel('bs')
plt.ylabel('time')
plt.title("blip_vqa_e2e_time")
plt.grid(True)
plt.legend()
plt.xticks(np.arange(x_min, x_max + 1, 1))
plt.savefig(root_path+"/"+"blip_vqa_e2e_time"+'.png') 



plt.figure()
plt.plot(x_data, [x_data[i]/y_sum[i] for i in range(len(y_sum))], label="3_sum")

x_data,y_data=prepare("blip_vqa_bls_e2e_time")
plt.plot(x_data, [x_data[i]/y_data[i] for i in range(len(y_data))], label="bls")

plt.xlabel('bs')
plt.ylabel('req/second')
plt.title("blip_vqa_e2e_throughput")
plt.grid(True)
plt.legend()
plt.xticks(np.arange(x_min, x_max + 1, 1))
plt.savefig(root_path+"/"+"blip_vqa_e2e_throughput"+'.png') 

