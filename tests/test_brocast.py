import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    dist.destroy_process_group()
def run(rank, size):
    """ Distributed function to be implemented. """
    # Set the GPU device
    torch.cuda.set_device(0)                                        
    torch.cuda.init()
    # Create a tensor on GPU
    tensor = torch.ones(1).cuda()
    dist.barrier()
    # All-reduce example (sum)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f'Rank {rank} has data {tensor[0]}')
def main():
    size = 4  # Number of processes
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
if __name__ == "__main__":
    main()

