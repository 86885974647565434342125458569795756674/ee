import time
import multiprocessing
from multiprocessing import shared_memory, Process
import numpy as np
import torch

def reader(shm_size):
    shm_name = 'my_shared_memory'
    shm = shared_memory.SharedMemory(name=shm_name)
    a=torch.as_tensor(np.copy(np.ndarray((shm_size), dtype=np.float32, buffer=shm.buf)),device="cuda")
    start_time = time.perf_counter()
    a=torch.as_tensor(np.copy(np.ndarray((shm_size), dtype=np.float32, buffer=shm.buf)),device="cuda")
    end_time = time.perf_counter()
    print(f"read time={end_time-start_time} {a.device}")
    shm.close()
    print("reader finish")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    shm_name = 'my_shared_memory'
    bses=[4,8,16,32]
    seq=1024
    hid=1024
    for bs in bses:
        print(f"bs={bs}")
        shm_size=bs*seq*hid
        shm = shared_memory.SharedMemory(name=shm_name,create=True,size=shm_size*np.dtype(np.float32).itemsize)
        b = np.random.random((shm_size)).astype(np.float32)

        a=torch.as_tensor(np.copy(np.ndarray((shm_size), dtype=np.float32, buffer=shm.buf)),device="cuda")

        start_time = time.perf_counter()
        a=torch.as_tensor(np.copy(np.ndarray((shm_size), dtype=np.float32, buffer=shm.buf)),device="cuda")
        end_time = time.perf_counter()
        print(f"write shm time={end_time-start_time} {a.device}")

        reader_process = Process(target=reader,args=(shm_size,))
        reader_process.start()
        reader_process.join()
        shm.close()
        shm.unlink()
        print("writer finish")

