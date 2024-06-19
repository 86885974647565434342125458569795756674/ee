import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

model_name = "blip_nlvr"

with httpclient.InferenceServerClient("localhost:8000") as client:
    images0=np.array([b"/workspace/demos/images/ex0_0.jpg",b"/workspace/demos/images/acorns_1.jpg"])
    images1=np.array([b"/workspace/demos/images/ex0_1.jpg",b"/workspace/demos/images/acorns_6.jpg"])


    questions = np.array(
        [b"The left image contains twice the number of dogs as the right image, and at least two dogs in total are standing.",
        b"One image shows exactly two brown acorns in back-to-back caps on green foliage.",]
    )

    inputs = [
        httpclient.InferInput(
            "INPUT0",
            images0.shape,
            np_to_triton_dtype(images0.dtype),
        ),
        httpclient.InferInput(
            "INPUT1",
            images1.shape,
            np_to_triton_dtype(images1.dtype),
        ),
        httpclient.InferInput(
            "INPUT2",
            questions.shape,
            np_to_triton_dtype(questions.dtype),
        ),
    ]

    inputs[0].set_data_from_numpy(images0)
    inputs[1].set_data_from_numpy(images1)
    inputs[2].set_data_from_numpy(questions)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    answers = response.as_numpy("OUTPUT0")

    print("IMAGE0 ({}) + IMAGE1 ({}) + QUESTION ({}) = ANSWER ({})".format(
        images0,images1, questions, answers))
