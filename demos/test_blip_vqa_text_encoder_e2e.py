import numpy as np
import time
from tritonclient.utils import *
import tritonclient.http as httpclient
import sys

root_path='/dynamic_batch/triton-multi-modal-serving'
model_name = "blip_vqa_text_encoder"

conn_timeout_second=60*100
client_wait_ms=1000000*60*100

bs=1
if len(sys.argv) > 1:
    bs = int(sys.argv[1])
print(f"bs={bs}")

input0_data = np.load(root_path+"/pretrained/images_embeds.npy")
input0_data = np.repeat(input0_data,bs,axis=0)
input1_data = np.array([[b"where is the woman sitting?"]])
input1_data = np.repeat(input1_data,bs,axis=0)

client=httpclient.InferenceServerClient("localhost:8000")

inputs=[
    httpclient.InferInput(
        "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
    ),
    httpclient.InferInput(
        "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
    ),
]


inputs[0].set_data_from_numpy(input0_data)
inputs[1].set_data_from_numpy(input1_data)

outputs=[
            httpclient.InferRequestedOutput("OUTPUT0"),
                ]

response=client.infer(model_name, inputs, outputs=outputs, timeout=client_wait_ms)

client.close()


client=httpclient.InferenceServerClient("localhost:8000")

inputs=[
    httpclient.InferInput(
        "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
    ),
    httpclient.InferInput(
        "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
    ),
]


inputs[0].set_data_from_numpy(input0_data)
inputs[1].set_data_from_numpy(input1_data)

outputs=[
            httpclient.InferRequestedOutput("OUTPUT0"),
                ]

start_time=time.time()
response=client.infer(model_name, inputs, outputs=outputs, timeout=client_wait_ms)
end_time=time.time()
print(f"time={end_time-start_time}")

with open(root_path+"/blip_vqa_text_encoder_e2e_time.txt","a") as f:
            f.write(f"{bs},{end_time-start_time}\n")

client.close()

