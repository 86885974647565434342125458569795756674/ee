import torch
import torch.multiprocessing as mp

class Engine:
    def __init__(self,name,model_creater,model_kwargs,control_conn,pre_conn,post_conn) -> None:
        self.name=name
        self.model=model_creater(**model_kwargs)
        self.model.eval()
        self.control_conn=control_conn
        self.pre_conn=pre_conn
        self.post_conn=post_conn
        print(f"start {self.name}")
    def loop(self):
        with torch.no_grad():
            while True:
                input_data=self.pre_conn.recv()
                print(f"size of {self.name}={len(input_data)}")
                self.post_conn.send(self.model(input_data))        

def run_engine(name,model_creater,model_kwargs,control_conn,pre_conn,post_conn):
    e=Engine(name,model_creater,model_kwargs,control_conn,pre_conn,post_conn)
    e.loop()

if __name__=="__main__":
    import sys

    root_path='/dynamic_batch/ee/'

    sys.path.append(root_path)
    from models.blip.blip_vqa_visual_encoder import blip_vqa_visual_encoder
    from models.blip.blip_vqa_text_encoder import blip_vqa_text_encoder
    from models.blip.blip_vqa_text_decoder import blip_vqa_text_decoder

    model_url = root_path+"pretrained/model_base_vqa_capfilt_large.pth"
    kwargs={"pretrained":model_url, "vit":"base"}

    # contorl_conn
    c0,c1=mp.Pipe()
    # data_conn
    d0,d1=mp.Pipe()
    d2,d3=mp.Pipe()
    engine_process = mp.Process(target=run_engine, args=("blip_vqa_visual_encoder",blip_vqa_visual_encoder,kwargs,c1,d1,d2,))
    engine_process.start()

    bs=10
    data=[root_path.encode('utf-8')+b"/demos/images/merlion.png"]*bs
    d0.send(data)
    data=d3.recv()
    print(data.shape,data.device)
    print("finish............")

    engine_process.join()
