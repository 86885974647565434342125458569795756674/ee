import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

model_name = "blip_itm"
loop_size = 4

with httpclient.InferenceServerClient("localhost:8000") as client:
    image_urls = np.array([
        b"/workspace/demos/images/beach.jpg",
    ] * loop_size)

    captions = np.array([
        b'a woman sitting on the beach with a dog',
    ] * loop_size)

    inputs = [
        httpclient.InferInput(
            "IMAGE",
            image_urls.shape,
            np_to_triton_dtype(image_urls.dtype),
        ),
        httpclient.InferInput(
            "CAPTION",
            captions.shape,
            np_to_triton_dtype(captions.dtype),
        ),
    ]

    inputs[0].set_data_from_numpy(image_urls)
    inputs[1].set_data_from_numpy(captions)

    outputs = [
        httpclient.InferRequestedOutput("SCORE"),
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    scores = response.as_numpy("SCORE")

    print("IMAGE ({}) + CAPTION ({})= SCORE ({})".format(
        image_urls, captions, scores))

    sys.exit(0)
