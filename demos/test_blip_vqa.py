import sys
import time
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

root_path='/dynamic_batch/triton-multi-modal-serving'

model_name = "blip_vqa"
batch_size = 32

with httpclient.InferenceServerClient("localhost:8000") as client:
    start_time=time.time()

    images = np.array([
        root_path.encode('utf-8')+b"/demos/images/beach.jpg"] * batch_size)
    questions = np.array([
        b"where is the woman sitting?"] * batch_size)
    inputs = [
        httpclient.InferInput(
            "INPUT0",
            images.shape,
            np_to_triton_dtype(images.dtype),
        ),
        httpclient.InferInput(
            "INPUT1",
            questions.shape,
            np_to_triton_dtype(questions.dtype),
        ),
    ]

    inputs[0].set_data_from_numpy(images)
    inputs[1].set_data_from_numpy(questions)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    answers = response.as_numpy("OUTPUT0")

    print("client:",start_time,time.time())
    
    print("IMAGE ({}) + QUESTION ({}) = ANSWER ({})".format(
        images.shape, questions.shape, answers.shape))
