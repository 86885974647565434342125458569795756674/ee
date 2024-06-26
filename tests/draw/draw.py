import matplotlib.pyplot as plt
import numpy as np
import sys

# Initialize lists
bs = []
open_list = []
image_preprocess = []
visual_encoder = []
text_preprocess = []
text_encoder = []
text_decoder = []

root_path="/dynamic_batch/ee/"
with open(root_path+"tests/blip_vqa_forward_time.txt", 'r') as file:
    data = file.read()

# Split the data into lines and process
lines = data.split('\n')

for line in lines:
    line = line.strip()

    if line.startswith("bs="):
        bs_value = int(line.split('=')[1])
        if bs_value not in bs:
            bs.append(bs_value)
    elif line.startswith("open:"):
        open_list.append(float(line.split(':')[1]))
    elif line.startswith("image preprocess:"):
        image_preprocess.append(float(line.split(':')[1]))
    elif line.startswith("visual_encoder time:"):
        visual_encoder.append(float(line.split(':')[1]))
    elif line.startswith("text_preprocess time:"):
        text_preprocess.append(float(line.split(':')[1]))
    elif line.startswith("text_encoder time:"):
        text_encoder.append(float(line.split(':')[1]))
    elif line.startswith("text_decoder time:"):
        text_decoder.append(float(line.split(':')[1]))

values1=[open_list[i]+image_preprocess[i] +visual_encoder[i] for i in range(len(bs))]
values2=[text_preprocess[i]+text_encoder[i] for i in range(len(bs))]
values3=text_decoder

plt.plot(bs, [bs[i]/values1[i] for i in range(len(bs))],label="visual_encoder")
plt.plot(bs, [bs[i]/values2[i] for i in range(len(bs))],label="text_encoder")
plt.plot(bs, [bs[i]/values3[i] for i in range(len(bs))],label="text_decoder")
plt.xlabel('bs')
plt.ylabel('throughput')
plt.title("blip_vqa_throughput")
plt.grid(True)
plt.legend()
x_min, x_max = min(bs), max(bs)
plt.xticks(np.arange(x_min, x_max + 1, 1))
plt.savefig('blip_vqa_throughput.png') 
