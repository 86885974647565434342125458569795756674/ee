import matplotlib.pyplot as plt
import numpy as np
import sys

root_path="/dynamic_batch/triton-multi-modal-serving"
file_name=sys.argv[1]+"_time"

x_data=[]
y_data=[]
with open(root_path+'/'+file_name+'.txt', 'r') as f:
    for line in f:
        x, y = line.strip().split(',')
        x_data.append(int(x))
        y_data.append(float(y))

plt.plot(x_data, y_data, 'bo-')
plt.xlabel('bs')
plt.ylabel('time')
plt.title(file_name)
plt.grid(True)
x_min, x_max = min(x_data), max(x_data)
plt.xticks(np.arange(x_min, x_max + 1, 1))
plt.savefig(root_path+"/"+file_name+'.png') 
