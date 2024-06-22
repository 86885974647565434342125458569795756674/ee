from multiprocessing import Lock
from multiprocessing.managers import SyncManager

class LockManager(SyncManager): pass

def get_lock():
    return Lock()

if __name__ == "__main__":
    LockManager.register("Lock", get_lock)
    manager = LockManager(address=("localhost", 50000), authkey=b'secret')
    server = manager.get_server()
    print("Lock server started")
    server.serve_forever()
