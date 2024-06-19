import numpy as np
import random
import time
import multiprocessing
import json
import os
from blip_vqa_process import while_client

if __name__ == "__main__":  
    try:
        # Create a list to hold process objects
        processes = []
        input_queue=multiprocessing.Queue()
        output_queue=multiprocessing.Queue()
        blip_vqa_process = multiprocessing.Process(target=while_client,args=("blip_vqa",input_queue,output_queue))
        processes.append(blip_vqa_process)
        blip_vqa_process.start()

        dataset_dir = "/workspace/datasets/vqa/"
        json_file = "/workspace/datasets/vqa/vqa_test.json"
        with open(json_file) as f:
            dataset = json.load(f)
        batch_size=1
        
        for request_num in [2,4,8,16,32,64,128]:
            print(f"\nrequest_num: {request_num}")
            for repeat_time in range(3):
                datas_list=[dataset[i*batch_size:(i+1)*batch_size] for i in range(request_num)]
                images_batches = [np.array([bytes(os.path.join(dataset_dir, data["image"]), "utf-8") for data in datas]) for datas in datas_list]
                texts_batches = [np.array([bytes(os.path.join(dataset_dir, data["question"]), "utf-8") for data in datas]) for datas in datas_list]
                livings_batches = [np.array([-1 for data in datas]) for datas in datas_list]

                request_ids, batch_nums,images,texts,livings=[],[],[],[],[]
                for i in range(request_num):
                    request_id,image_batch,text_batch,living_batch=i,images_batches[i],texts_batches[i],livings_batches[i]
                    request_ids.append(request_id)
                    images.append(image_batch)
                    texts.append(text_batch)
                    livings.append(living_batch)
                    batch_nums.append(image_batch.shape[0])        
                
                start_time=time.time()

                images=np.concatenate(images, axis=0)
                texts=np.concatenate(texts, axis=0)
                
                input_queue.put((images,texts))

                result = output_queue.get()
                end_time=time.time()
                if repeat_time==2:
                    print(f"total time: {end_time-start_time}")

        print("--------------------------------------------------------------------------")
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.join()
        print("finish...")
