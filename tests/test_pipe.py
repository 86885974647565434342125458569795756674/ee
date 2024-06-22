import torch
import torch.multiprocessing as mp
import time
import numpy as np
def psender(queue,pp,bs):
    # 创建一个tensor对象
    tensor_data = torch.zeros((64,1024,1024),dtype=torch.float32).to("cuda")
    #tensor_data = np.zeros((64,1024,1024),dtype=np.float32)
    # 发送tensor到队列
    #pp.send(tensor_data)
    queue.put(tensor_data)

    tensor_data = torch.zeros((bs,1024,1024),dtype=torch.float32).to("cuda")
    tensor_data_ = torch.zeros((bs,1024,1024),dtype=torch.float32).to("cuda")
    '''
    tensor_data = np.zeros((bs,1024,1024),dtype=np.float32)
    tensor_data_ = np.zeros((bs,1024,1024),dtype=np.float32)
    '''
    pp.recv()
    print(time.perf_counter())
    #pp.send((tensor_data,tensor_data_))
    queue.put([tensor_data,tensor_data_])
    pp.recv()
    #print(tensor_data[0][0][2])
    #print("send exit")

def preceiver(queue,cp):
    # 从队列中获取数据
    #received_data = cp.recv()
    received_data=queue.get()
    cp.send(1)
    #received_data = cp.recv()#.clone()# it is share without clone
    received_data=queue.get()
    received_data[0],received_data[1]=received_data[0].clone(),received_data[1].clone()
    #print(f"{time.perf_counter()} {received_data[1].shape} {received_data[0].device}")
    print(f"{time.perf_counter()} {received_data[1].shape} ")
    #received_data[0][0][2]=4
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
        #0.002s
        
        '''
        sender_process = mp.Process(target=sender, args=(queue,pp))
        receiver_process = mp.Process(target=receiver, args=(queue,cp))
        #0.002s
        '''
        sender_process.start()
        receiver_process.start()
        
        sender_process.join()
        receiver_process.join()
    '''
    numpy

    7026087.162516628
    7026087.387300556 (4, 1024, 1024)
    7026088.645324079
    7026088.989760245 (8, 1024, 1024)
    7026090.208292822
    7026090.873023893 (16, 1024, 1024)
    7026092.142630592
    7026093.350697263 (32, 1024, 1024)
    '''
