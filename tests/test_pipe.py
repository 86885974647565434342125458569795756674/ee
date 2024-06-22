import torch
import torch.multiprocessing as mp
import time

def sender(queue,pp):
    # 创建一个tensor对象
    tensor_data = torch.zeros((64,1024,1024),dtype=torch.float32).to("cuda")
    # 发送tensor到队列
    queue.put(tensor_data)

    tensor_data = torch.zeros((32,1024,1024),dtype=torch.float32).to("cuda")
    pp.recv()
    print(time.time())
    queue.put(tensor_data)
    pp.recv()

def receiver(queue,cp):
    # 从队列中获取数据
    received_data = queue.get()
    cp.send(1)
    received_data = queue.get()
    print(f"{time.time()} {received_data.shape} {received_data.device}")
    cp.send(1)

def psender(queue,pp,bs):
    # 创建一个tensor对象
    tensor_data = torch.zeros((64,1024,1024),dtype=torch.float32).to("cuda")
    # 发送tensor到队列
    pp.send(tensor_data)

    tensor_data = torch.zeros((bs,1024,1024),dtype=torch.float32).to("cuda")
    pp.recv()
    print(time.perf_counter())
    pp.send(tensor_data)
    pp.recv()
    #print(tensor_data[0][0][2])
    #print("send exit")

def preceiver(queue,cp):
    # 从队列中获取数据
    received_data = cp.recv()
    cp.send(1)
    received_data = cp.recv().clone()# it is share without clone
    print(f"{time.perf_counter()} {received_data.shape} {received_data.device}")
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
