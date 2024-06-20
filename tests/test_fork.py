import multiprocessing
import torch

def work():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.randn(2, 2, device=device)
    
if __name__=="__main__":
    processes=[]
    for _ in range(2):
        work_process=multiprocessing.Process(target=work)
        work_process.start()
        processes.append(work_process)
    for p in processes:
        p.join()
