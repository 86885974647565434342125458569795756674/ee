import http.server
import queue
import time
import numpy as np
import json
import random
import multiprocessing
import sys
import os
from blip_vqa_process import blip_vqa_process
from utils import change_batch_size

def send_and_receive(request_queue,request_events,processed_results):
    try:
        # write_file="/workspace/send_and_receive.txt"
        # if(os.path.isfile(write_file)):    
        #     os.remove(write_file)

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
                livings_batches = [np.array([1000 for data in datas]) for datas in datas_list]

                request_ids, batch_nums,images,texts,livings=[],[],[],[],[]
                for i in range(request_num):
                    request_id,image_batch,text_batch,living_batch=i,images_batches[i],texts_batches[i],livings_batches[i]
                    request_ids.append(request_id)
                    images.append(image_batch)
                    texts.append(text_batch)
                    livings.append(living_batch)
                    batch_nums.append(image_batch.shape[0])        

                # with open(write_file,"a") as f:
                #     f.write(f"request_num: {request_num}\n")
                #     f.write(str(batch_size_list)+"\n")
                
                start_time=time.time()
                request_queue.put((request_ids,batch_nums,images,texts,livings))

                results=[]
                end_times=[]
                request_count=0
                while request_count<request_num:
                    result = processed_results.get(request_count,None)
                    if result is not None:
                        end_times.append(time.time())
                        # results.append(result)
                        request_count+=1
                
                for request_count in range(request_num):
                    del processed_results[request_count]

                # for e in end_times:
                #     print(e-start_time)
                if repeat_time==2:
                    print(f"total time: {end_times[-1]-start_time}")
                    print(f"avg time: {sum(end_times)/len(end_times)-start_time}")

                # for r in results:
                #     print(r)

                # with open(write_file,"a") as f:
                #     for i in results:
                #         f.write(str(i)+"\n")
        print("--------------------------------------------------------------------------")
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            fix_batch = (int(x) for x in sys.argv[1].split(','))

        # Create a list to hold process objects
        processes = []

        batch_size_queue=multiprocessing.Queue()
        # queue is not the fastest way, maybe pipe
        time_interval=1
        change_batch_size_process = multiprocessing.Process(target=change_batch_size, args=(batch_size_queue,time_interval,3))
        processes.append(change_batch_size_process)
        change_batch_size_process.start()

        request_queue=multiprocessing.Queue()
        manager = multiprocessing.Manager()
        request_events=manager.dict()
        processed_results = manager.dict()

        blip_vqa_process_process = multiprocessing.Process(target=blip_vqa_process, args=(request_queue,request_events,processed_results,batch_size_queue,fix_batch))
        processes.append(blip_vqa_process_process)
        blip_vqa_process_process.start()
        
        send_and_receive_process = multiprocessing.Process(target=send_and_receive, args=(request_queue,request_events,processed_results,))
        processes.append(send_and_receive_process)
        send_and_receive_process.start()
        
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.join()
        print("finish...")
