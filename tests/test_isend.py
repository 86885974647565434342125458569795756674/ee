import os
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
import torch

def model_worker1(RANK,WORLD_SIZE,lock):
    os.environ['MASTER_PORT']='12345'
    os.environ['MASTER_ADDR']='localhost'
    os.environ['WORLD_SIZE']=str(WORLD_SIZE)
    os.environ['RANK']=str(RANK)

    dist.init_process_group("nccl",rank=RANK,world_size=WORLD_SIZE)
    send_tensor = torch.arange(2, dtype=torch.float32).to(f'cuda:0') + 2 * RANK
    recv_tensor = torch.randn(2, dtype=torch.float32).to(f'cuda:0')
    send_op = dist.P2POp(dist.isend, send_tensor, (RANK + 1)%WORLD_SIZE)
    recv_op = dist.P2POp(dist.irecv, recv_tensor, (RANK - 1 + WORLD_SIZE)%WORLD_SIZE)
    reqs = dist.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    print(recv_tensor)

def model_worker(RANK,WORLD_SIZE,lock):
    os.environ['MASTER_PORT']='12345'
    os.environ['MASTER_ADDR']='localhost'
    os.environ['WORLD_SIZE']=str(WORLD_SIZE)
    os.environ['RANK']=str(RANK)

    dist.init_process_group("nccl",rank=RANK,world_size=WORLD_SIZE)
    sended_tensor=torch.zeros((2,2),dtype=torch.float).to("cuda:0")+RANK
    received_tensor=torch.empty((2,2),dtype=torch.float).to("cuda:0")
    send_stream = torch.cuda.Stream()
    receive_stream = torch.cuda.Stream()
    send_handle = None
    recv_handle = None

    print("///")
    print(RANK)
    with torch.cuda.stream(send_stream):
        send_handle = dist.isend(sended_tensor, (RANK + 1) % WORLD_SIZE)
    print("....")
    with torch.cuda.stream(receive_stream):
        recv_handle = dist.irecv(received_tensor, (RANK - 1 + WORLD_SIZE) % WORLD_SIZE)
    # Ensure that the communication operations are complete
    print("???")
    torch.cuda.synchronize()

if __name__=="__main__":
    WORLD_SIZE=2
    lock=multiprocessing.Lock()
    processes=[]
    for i in range(WORLD_SIZE):
        process = multiprocessing.Process(target=model_worker1, args=(i,WORLD_SIZE,lock))
        processes.append(process)
        process.start()
    for p in processes:
        p.join()
