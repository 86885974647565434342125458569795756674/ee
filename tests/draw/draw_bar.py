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

values1=open_list
values2=[open_list[i]+image_preprocess[i] for i in range(len(bs))] 
values3=[open_list[i]+image_preprocess[i]+visual_encoder[i] for i in range(len(bs))]
values4=text_preprocess
values5=[text_preprocess[i]+text_encoder[i] for i in range(len(bs))]
values6=text_decoder

bar_width = 1/7

r1 = np.arange(len(bs))
r2 = [x + bar_width for x in r1]
r3=[x+bar_width for x in r2]
r4=[x+bar_width for x in r3]
r5=[x+bar_width for x in r4]
r6=[x+bar_width for x in r5]

plt.bar(r1, values1, color='blue', width=bar_width, edgecolor='grey', label='image_open')
plt.bar(r2, values2, color='green', width=bar_width, edgecolor='grey', label='image_open+image_preprocess')
plt.bar(r3, values3, color='red', width=bar_width, edgecolor='grey', label='image_open+image_preprocess_visual_encoder')
plt.bar(r4, values4, color='orange', width=bar_width, edgecolor='grey', label='text_preprocess')
plt.bar(r5, values5, color='purple', width=bar_width, edgecolor='grey', label='text_preprocess+text_encoder')
plt.bar(r6, values6, color='cyan', width=bar_width, edgecolor='grey', label='text_decoder')

plt.xlabel('bs')
plt.ylabel('time')
plt.title('blip_vqa')
plt.xticks([r + 2.5 * bar_width for r in r1], bs)

plt.legend()

plt.savefig("blip_vqa_bar.png") 
