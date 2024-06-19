import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

model_name = "blip_nlvr_text_encoder"

with httpclient.InferenceServerClient("localhost:8000") as client:
    images_embeds = np.load("/workspace/pretrained/blip_nlvr_images_embeds.npy")

    questions = np.array(
        [b"The left image contains twice the number of dogs as the right image, and at least two dogs in total are standing.",
        b"One image shows exactly two brown acorns in back-to-back caps on green foliage.",]
    )

    inputs = [
        httpclient.InferInput(
            "INPUT0",
            images_embeds.shape,
            np_to_triton_dtype(images_embeds.dtype),
        ),
        httpclient.InferInput(
            "INPUT1",
            questions.shape,
            np_to_triton_dtype(questions.dtype),
        ),
    ]

    inputs[0].set_data_from_numpy(images_embeds)
    inputs[1].set_data_from_numpy(questions)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    answers = response.as_numpy("OUTPUT0")

    print("IMAGE ({}) + QUESTION ({}) = ANSWER ({})".format(
        images_embeds.shape, questions, answers))
