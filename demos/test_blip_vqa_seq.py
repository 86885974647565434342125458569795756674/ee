import sys
import torch
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *
import time

batch_size=32

with httpclient.InferenceServerClient("localhost:8000") as client:
    start_time=time.time()
    
    # Visual Encoder
    input0_data = np.array([b"/workspace/demos/images/beach.jpg"] * batch_size)

    inputs = [
        httpclient.InferInput(
            "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(input0_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    response = client.infer("blip_vqa_visual_encoder", inputs, request_id=str(1), outputs=outputs)

    output0_data = response.as_numpy("OUTPUT0")

    # print("INPUT0 ({})  = OUTPUT0 ({})".format(input0_data.shape, output0_data.shape))

    visual_time=time.time()

    # Text Encoder
    input0_data = output0_data
    input1_data = np.array(
        [b"where is the woman sitting?"]*batch_size
    )

    inputs = [
        httpclient.InferInput(
            "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
        httpclient.InferInput(
            "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    response = client.infer(
        "blip_vqa_text_encoder", inputs, request_id=str(1), outputs=outputs
    )

    output0_data = response.as_numpy("OUTPUT0")

    # print(
    #     "INPUT0 {} + INPUT1 {} = OUTPUT0 {}".format(
    #         input0_data.shape, input1_data, output0_data.shape
    #     )
    # )

    text_encoder_time=time.time()
    # Text Decoder
    input0_data = output0_data

    input1_data_shape=(input0_data.shape[0]*input0_data.shape[1],input0_data.shape[2])
    input1_data = torch.ones(input1_data_shape, dtype=torch.long).numpy(force=True)

    inputs = [
        httpclient.InferInput(
            "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
        httpclient.InferInput(
            "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)


    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    response = client.infer(
        "blip_vqa_text_decoder", inputs, request_id=str(1), outputs=outputs
    )

    result = response.get_response()

    output0_data = response.as_numpy("OUTPUT0")

    print("client:",start_time,visual_time,text_encoder_time,time.time())
    
    print("INPUT0 {} INPUT1 {} = OUTPUT0 {}".format(input0_data.shape,input1_data.shape,output0_data.shape))
