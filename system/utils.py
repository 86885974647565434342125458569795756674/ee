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

def while_client(model_name,input_queue,output_queue):
    try:
        with httpclient.InferenceServerClient("localhost:8000") as client:
            while True:
                input_data=input_queue.get()
                inputs=[]
                if isinstance(input_data,tuple):
                    for i in range(len(input_data)):
                        inputs.append(
                            httpclient.InferInput(
                            f"INPUT{i}", input_data[i].shape, np_to_triton_dtype(input_data[i].dtype)
                            )
                        )
                    for i in range(len(inputs)):
                        inputs[i].set_data_from_numpy(input_data[i])
                else:
                    inputs.append(
                        httpclient.InferInput(
                        f"INPUT0", input_data.shape, np_to_triton_dtype(input_data.dtype)
                        )
                    )
                    inputs[0].set_data_from_numpy(input_data)

                outputs = [
                    httpclient.InferRequestedOutput("OUTPUT0"),
                ]

                response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

                output0_data = response.as_numpy("OUTPUT0")
                output_queue.put(output0_data,block=False)
    except KeyboardInterrupt:
        pass

def change_batch_size(batch_size_queue,time_interval,num_batch):
    try:
        while True:
            batch_size_queue.put((random.randint(1, 10) for _ in range(num_batch)),block=False)
            # print(f"release batch size: {a},{b},{c}")
            time.sleep(time_interval)
    except KeyboardInterrupt:
        pass