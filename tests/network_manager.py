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
    data = b'New Data'

    shm = shared_memory.SharedMemory(name=shared_mem_name,create=True,size=1024)
    try:
        lock.acquire()
        try:
            print(f"Process {os.getpid()} writing to shared memory")
            print(type(shm.buf))
            for i in range(len(data)):
                shm.buf[i] = data[i]
            print(len(shm.buf))
        finally:
            lock.release()
        time.sleep(2)  # Simulate a write delay
        while True:
            pass
    except KeyboardInterrupt:
        shm.close()
        shm.unlink()
        print("writer quit")
