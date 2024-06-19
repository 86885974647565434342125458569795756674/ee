import numpy as np
import time
from tritonclient.utils import *
import numpy as np
import asyncio
import tritonclient.http.aio as httpclient

root_path='/dynamic_batch/triton-multi-modal-serving'
model_name = "blip_vqa_text_decoder"

input0_data = np.load(root_path+"/pretrained/questions_states.npy")
# input1_data_shape=(input0_data.shape[0]*input0_data.shape[1],input0_data.shape[2])
# input1_data = torch.ones(input1_data_shape, dtype=torch.long).numpy(force=True)

clients=[]
inputss=[]
outputss=[]
responses=[]
times=[]

total_second=5
user_num=32
conn_timeout_second=60*100
client_wait_ms=1000000*60*100

async def infer_async(sleep_time, client, inputs, outputs):
    await asyncio.sleep(sleep_time)
    response=await client.infer(model_name, inputs, outputs=outputs, timeout=client_wait_ms)
    return response,time.time()

async def main_async():
    for i in range(total_second):
        for _ in range(user_num):
            clients.append(httpclient.InferenceServerClient("localhost:8000",conn_timeout=conn_timeout_second))

            inputss.append([
                httpclient.InferInput(
                    "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
                ),
                # httpclient.InferInput(
                #     "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
                # ),
            ])


            inputss[-1][0].set_data_from_numpy(input0_data)
            # inputss[-1][1].set_data_from_numpy(input1_data)

            outputss.append([
                httpclient.InferRequestedOutput("OUTPUT0"),
                ])

    tasks = [infer_async(t,clients[t*user_num+u], inputss[t*user_num+u], outputss[t*user_num+u]) for t in range(total_second) for u in range(user_num)]
    responses_times = await asyncio.gather(*tasks)
    for rt in responses_times:
        responses.append(rt[0])
        times.append(rt[1])

    for client in clients:
        await client.close()

if __name__=="__main__":
    print(user_num*total_second)
    print(f"start client:{time.time()}")
    asyncio.run(main_async())
    print(f"end client:{time.time()}")

    output0_datas = np.concatenate([response.as_numpy("OUTPUT0") for response in responses],axis=0)
    print("OUTPUT0 {}".format(output0_datas.shape))
    #times.sort()
    #for t in times:
     #   print(t)
    
