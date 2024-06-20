import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import tritonclient.http as httpclient
from tritonclient.utils import *
import time
import multiprocessing
import queue
import random
import os

def change_batch_size(batch_size_queue,time_interval,num_batch):
    try:
        while True:
            batch_size_queue.put([random.randint(1, 10) for _ in range(num_batch)],block=False)
            # print(f"release batch size: {a},{b},{c}")
            time.sleep(time_interval)
    except KeyboardInterrupt:
        pass
