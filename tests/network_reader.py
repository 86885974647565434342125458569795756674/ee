import time
import os
from multiprocessing import shared_memory
from multiprocessing.managers import SyncManager

class LockManager(SyncManager): pass

if __name__ == "__main__":
    LockManager.register("Lock")
    manager = LockManager(address=("localhost", 50000), authkey=b'secret')
    manager.connect()
    lock = manager.Lock()

    shared_mem_name = 'my_shared_memory'

    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
    try:
        lock.acquire()
        try:
            print(f"Process {os.getpid()} reading from shared memory")
            data = bytes(existing_shm.buf[:1024])  # Assuming the shared memory size is 1024
        finally:
            lock.release()
            time.sleep(2)  # Simulate a read delay
        print(f"Read data: {data}")
        while True:
            pass
    except KeyboardInterrupt:
        existing_shm.close()
        print("reader quit")

