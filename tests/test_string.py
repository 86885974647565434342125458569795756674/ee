import torch
import torch.multiprocessing as mp
import time
import numpy as np

root_path='/dynamic_batch/ee/'

def psender(queue,pp,bs):
    # 创建一个tensor对象
    tensor_data=[root_path.encode('utf-8')+b"/demos/images/merlion.png"]*bs
    tensor_data=np.array([root_path.encode('utf-8')+b"/demos/images/merlion.png"]*bs)
    # 发送tensor到队列
#    pp.send(tensor_data)
    queue.put(tensor_data)

    tensor_data=[root_path.encode('utf-8')+b"/demos/images/merlion.png"]*bs
    tensor_data=np.array([root_path.encode('utf-8')+b"/demos/images/merlion.png"]*bs)
    pp.recv()
    print(time.perf_counter())
    #pp.send(tensor_data)
    queue.put(tensor_data)
    pp.recv()
    #print(tensor_data[0][0][2])
    #print("send exit")

def preceiver(queue,cp):
    # 从队列中获取数据
    #received_data = cp.recv()
    received_data=queue.get()

    cp.send(1)
    #received_data = cp.recv()
    received_data=queue.get()
    print(f"{time.perf_counter()} {len(received_data)}")
    cp.send(1)

    #time.sleep(7)
    #print(received_data[0][0][2])
    #seem safe for sender to exit early

if __name__ == '__main__':
    # 初始化队列
    queue = mp.Queue()
    
    pp, cp = mp.Pipe()

    # 创建并启动发送和接收进程

    for bs in [4,8,16,32]:
        sender_process = mp.Process(target=psender, args=(queue,pp,bs))
        receiver_process = mp.Process(target=preceiver, args=(queue,cp))
        #0.0003s list
        #0.0005s numpy

        sender_process.start()
        receiver_process.start()
        
        sender_process.join()
        receiver_process.join()
