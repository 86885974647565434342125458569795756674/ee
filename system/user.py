import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import multiprocessing
import time
import os

# the number of events occurring within the specified interval.
# np.random.poisson(lam) lam is avg number
def poisson_request(lambd,total_time):
    np.random.seed(0)
    request_list=[]
    for _ in range(total_time):
        request_list.append(np.random.poisson(lambd))

    # print(max(request_list),min(request_list),sum(request_list)/len(request_list))
    # plt.bar(range(total_time)[:100], request_list[:100])
    # plt.show()

    return request_list

def azure_request(begin_time,end_time,acc):
    azure_2021 = pd.read_csv("/workspace/datasets/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt", sep=',')
    azure_2021["begin_timestamp"] = azure_2021["end_timestamp"] - azure_2021["duration"]
    
    azure_2021.drop(azure_2021[azure_2021.begin_timestamp < begin_time].index, inplace=True)
    azure_2021.drop(azure_2021[azure_2021.begin_timestamp > end_time].index, inplace=True)
    azure_2021.begin_timestamp = azure_2021.begin_timestamp.astype(int)

    request_list = [0]*((end_time + 1-begin_time)//acc)

    for indx, values in azure_2021['begin_timestamp'].value_counts().items():
        request_list[(indx-begin_time)//acc] += values

    # print(max(request_list),min(request_list),sum(request_list)/len(request_list))
    # plt.bar(range(begin_time,end_time+1)[:100], request_list[:100])
    # plt.show()

    return request_list

def generate_request_num(workload_name,acc=8,fix=32,lambd=32):
    day_second = 60 * 60 * 24
    begin_time = day_second * 3
    end_time = day_second * 4

    request_list=[]
    if workload_name=="fix":
        request_list=[fix]*((end_time-begin_time+1)//acc)
    elif workload_name=="poisson":
        request_list=poisson_request(lambd,(end_time-begin_time+1)//acc)
    elif workload_name=="azure":
        request_list=azure_request(begin_time,end_time,acc)

    return request_list

def send_request(dataset_dir,json_file,request_queue,workload_name,image_name,text_name):
    try:
        request_list=generate_request_num(workload_name)
        request_id=0
        request_id_max=len(request_list)-1

        with open("/workspace/send_request_list.txt","w") as f:
            f.write(str(request_list))

        with open(json_file) as f:
            dataset = json.load(f)
        dataset_id=0
        dataset_id_max=len(dataset)-1
        # print(request_id_max,sum(request_list),dataset_id_max)
        # 10799 345600 447792

        images_list=[]
        texts_list=[]
        livings_list=[]
        while request_id<=request_id_max and dataset_id<=dataset_id_max:
            datas=dataset[dataset_id:dataset_id+request_list[request_id]]
            # print(len(datas))
            dataset_id+=request_list[request_id]
            request_id+=1
            images_list.append(np.array([bytes(os.path.join(dataset_dir,data[image_name]),"utf-8") for data in datas]))
            texts_list.append(np.array([bytes(os.path.join(dataset_dir, data[text_name]), "utf-8") for data in datas]))
            livings_list.append(np.array([1000000]*len(datas)))

        batch_id=0
        batch_num=len(images_list)
        time_interval=1
        while batch_id<batch_num:
            request_queue.put((images_list[batch_id],texts_list[batch_id],livings_list[batch_id],time.time()),block=False)
            batch_id+=1
            time.sleep(time_interval)
        print("send_request quit")
    except KeyboardInterrupt:
        print("send_request quit")

def test_send_request():
    try:
        processes=[]

        dataset_dir = "/workspace/datasets/vqa/"
        json_file = "/workspace/datasets/vqa/vqa_test.json"

        request_queue=multiprocessing.Queue()
        processed_results = multiprocessing.Queue()
        
        send_request_process = multiprocessing.Process(target=send_request, args=(dataset_dir,json_file,request_queue,"azure","image","question",))
        processes.append(send_request_process)
        send_request_process.start()

        # receive_request_process = multiprocessing.Process(target=receive_request, args=(processed_results,))
        # processes.append(receive_request_process)
        # receive_request_process.start()

        latencies_list=[]
        while True:
            if not request_queue.empty():
                images_list=[]
                texts_list=[]
                livings_list=[]
                start_times_list=[]
                while not request_queue.empty():
                    images,texts,livings,start_time=request_queue.get()
                    images_list.append(images)
                    texts_list.append(texts)
                    livings_list.append(livings)
                    start_times_list.append(np.array([start_time]*images.shape[0]))
                images=np.concatenate(images_list,axis=0)
                texts=np.concatenate(texts_list,axis=0)
                livings=np.concatenate(livings_list,axis=0)   
                start_times=np.concatenate(start_times_list,axis=0)

                time.sleep(2)
                
                latencies_list.append(np.array([time.time()]*start_times.shape[0])-start_times)

    except KeyboardInterrupt:
        for latencies in latencies_list:
            print(latencies.shape[0])
            print(latencies)

        for process in processes:
            process.join()
        print("user quit")

def receive_request(processed_results):
    results_list=[]
    latencies_list=[]
    try:
        while True:
            results,start_times = processed_results.get()
            latencies_list.append(np.array([time.time()]*start_times.shape[0])-start_times)
            results_list.append(results)
    except KeyboardInterrupt:
        for latencies in latencies_list:
            print(latencies.shape[0])
            print(latencies)
        print("receive_request quit")

def test_receive_request():
    try:
        processes=[]

        dataset_dir = "/workspace/datasets/vqa/"
        json_file = "/workspace/datasets/vqa/vqa_test.json"

        request_queue=multiprocessing.Queue()
        processed_results = multiprocessing.Queue()
        
        send_request_process = multiprocessing.Process(target=send_request, args=(dataset_dir,json_file,request_queue,"azure","image","question",))
        processes.append(send_request_process)
        send_request_process.start()

        receive_request_process = multiprocessing.Process(target=receive_request, args=(processed_results,))
        processes.append(receive_request_process)
        receive_request_process.start()

        latencies_list=[]
        while True:
            if not request_queue.empty():
                images_list=[]
                texts_list=[]
                livings_list=[]
                start_times_list=[]
                while not request_queue.empty():
                    images,texts,livings,start_time=request_queue.get()
                    images_list.append(images)
                    texts_list.append(texts)
                    livings_list.append(livings)
                    start_times_list.append(np.array([start_time]*images.shape[0]))
                images=np.concatenate(images_list,axis=0)
                texts=np.concatenate(texts_list,axis=0)
                livings=np.concatenate(livings_list,axis=0)   
                start_times=np.concatenate(start_times_list,axis=0)

                time.sleep(2)
                
                processed_results.put((images,start_times),block=False)

    except KeyboardInterrupt:
        for latencies in latencies_list:
            print(latencies.shape[0])
            print(latencies)

        for process in processes:
            process.join()
        print("user quit")

def sent_queue_time(request_queue,shape):
    try:
        for _ in range(8):
            t=time.time()
            request_queue.put((np.empty(shape,dtype=np.float32),t))
        print("sent_queue_time quit")
    except KeyboardInterrupt:
        print("sent_queue_time quit")

def test_sent_queue_time():
    try:
        processes=[]

        request_queue=multiprocessing.Queue(maxsize=1)
        batch_size=32
        shape=(batch_size,901,768)
        print(shape)
        sent_queue_time_process = multiprocessing.Process(target=sent_queue_time, args=(request_queue,shape))
        processes.append(sent_queue_time_process)
        sent_queue_time_process.start()
        
        for _ in range(8):
            result,start_time=request_queue.get()
            t=time.time()
            print(f"latency:{t-start_time}")

        print("user quit")
    except KeyboardInterrupt:
        for process in processes:
            process.join()
        print("user quit")
        
if __name__=="__main__":
    # test_send_request()
    # test_receive_request()
    test_sent_queue_time()
