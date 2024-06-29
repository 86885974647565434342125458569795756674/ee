import http.server
import queue
import time
import numpy as np
import json
import random
import torch
import torch.multiprocessing as multiprocessing
import sys
import os

root_path="/dynamic_batch/ee/"
dataset_dir = root_path+"datasets/vqa/"

sys.path.append(root_path)
from system.blip.vqa.blip_vqa_process import blip_vqa_process
from system.utils import change_batch_size
from system.cache import cache_get_put,recover_cache_duplication,recover_once
from models.blip.blip_vqa_visual_encoder import blip_vqa_visual_encoder
from models.blip.blip_vqa_text_encoder import blip_vqa_text_encoder
from models.blip.blip_vqa_text_decoder import blip_vqa_text_decoder


class blip_vqa_request:
    def __init__(self,id,image,text,priority,cache):
        self.id=id

        self.image=image
        self.text=text
        # todo general content

        self.start_time=time.perf_counter()

        self.priority=priority
        # 0

        self.cache=cache
        # -1 no cache

    def set_time(self,time):
        self.start_time=time

def generate_input(input_queue,start_time_queue,wait_ready):
    try:
        time_slot=1

        json_file = dataset_dir+"vqa_test.json"

        with open(json_file) as f:
            dataset = json.load(f)

        request_num_list=[4,4,4,4]#[2,4,8,16,32]

        request_ids=list(range(request_num_list[0]))
        request_id_togethers=[list(range(request_num_list[0]))]

        i=1
        while i<len(request_num_list):
            request_id_togethers=request_id_togethers+[list(range(request_ids[-1]+1,request_ids[-1]+1+request_num_list[i]))]
            request_ids=request_ids+list(range(request_ids[-1]+1,request_ids[-1]+1+request_num_list[i]))
            i+=1
    
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]    
        # [[0, 1], [2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]]
        
        request_togethers=[]
        for id_togethers in request_id_togethers:
            tmp=[]
            for id in id_togethers:
                #image=bytes(os.path.join(dataset_dir, dataset[id]["image"]), "utf-8")
                image=bytes(os.path.join(dataset_dir, dataset[id]["image"].replace("test2015/", "test2015/re_")), "utf-8")
                text=bytes(dataset[id]["question"], "utf-8")
                tmp.append(blip_vqa_request(id,image,text,0,-1))
            request_togethers.append(tmp)

        # wait model warm up
        wait_time=wait_ready.get(block=True)
        for _ in range(wait_time):
            wait_ready.get(block=True)
        print(f"ready to send input.................................")

        start_time_queue.put(sum(request_num_list),block=True)
        for rs in request_togethers:
            start_time_dict={}
            for r in rs:
                start_time=time.perf_counter()
                r.set_time(start_time)
                start_time_dict[r.id]=start_time
            input_queue.put(rs,block=False)
            start_time_queue.put(start_time_dict,block=False)
            time.sleep(time_slot)
        print("generate_input finish....................................")
    except KeyboardInterrupt:
        print("generate_input exit....................................")
        
def record_output(out_queue,start_time_queue,bs_time,v_time):
    try:
        request_num=start_time_queue.get(block=True)

        # # start_time
        start_time_dict={}
        while len(start_time_dict)<request_num:
            start_time_dict.update(start_time_queue.get(block=True))

        # bs_time
        bs_time_dict={}
        while len(bs_time_dict)<request_num:
            b_t=bs_time[0].get(block=True)
            ids,end_time=b_t[0],b_t[1]
            bs_time_dict.update(dict(zip(ids,[end_time]*len(ids))))

        # v_time
        v_time_dict={}
        while len(v_time_dict)<request_num:
            v_t=v_time.get(block=True)
            ids,end_time=v_t[0],v_t[1]
            v_time_dict.update(dict(zip(ids,[end_time]*len(ids))))

            for i in ids:
                print(f"id={i},start_time={start_time_dict[i]},end_time={v_time_dict[i]},bs_time-start_time={bs_time_dict[i]-start_time_dict[i]},v_time-bs_time={v_time_dict[i]-bs_time_dict[i]},latency={v_time_dict[i]-start_time_dict[i]}")
            print()

        print("record_output finish..................................................")
    except KeyboardInterrupt:
        print("record_output exit..................................................")

def cache_manager(input_queue,cm2c_list,c2cm_list,cm2bs,):
    try:
        dict_list=[{}]*len(cm2c_list)
        while True:
            try:
                r_b=input_queue.get(block=False)
                # check cache using dict_list and mark r_b
                ###########################
                cm2bs.put(r_b,block=False)
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        print("cache_manager exit..................................................")

def bach_scheduler(cm2bs,bs2e_list,bs_policy,bs_time):
    try:
        # algo need what data struct?
        # append req directly for now
        if bs_policy["kind"]=="fix":
            image_list=[]
            image_ids=[]
            text_list=[]
            text_ids=[]
            bs_list=[]
            bs_list=bs_list+[int(bs) for bs in bs_policy["bses"].split(",")]
            # print(f"bs_list={bs_list}")
        while True:    
            # algo to generate batches
            # append req directly for now
            if bs_policy["kind"]=="fix":
                try:
                    r_b=cm2bs.get(block=False)
                    for r in r_b:
                        if r.image is not None:
                            image_list.append(r.image)
                            image_ids.append(r.id)
                        if r.text is not None:
                            text_list.append(r.text)
                            text_ids.append(r.id)
                except queue.Empty:
                    pass
                if len(image_list)!=0:
                    # print(f"len(image_list)={len(image_list)}")
                    i=0
                    while i+bs_list[0]<=len(image_list):
                        bs2e_list[0].put([image_ids[i:i+bs_list[0]],image_list[i:i+bs_list[0]]],block=False)
                        bs_time[0].put([image_ids[i:i+bs_list[0]],time.perf_counter()],block=False)
                        i+=bs_list[0]
                    if i<len(image_list):
                        bs2e_list[0].put([image_ids[i:],image_list[i:]],block=False)
                        bs_time[0].put([image_ids[i:],time.perf_counter()],block=False)
                    image_list=[]
                    image_ids=[]
    except KeyboardInterrupt:
        print("bach_scheduler exit..................................................")

def blip_vqa_visual_encoder_engine(c2v,v2c,pre_queue,bs2ve,output_queue,v_time,wait_ready):
    try:
        model_url = root_path+"pretrained/model_base_vqa_capfilt_large.pth"
        model = blip_vqa_visual_encoder(pretrained=model_url, vit="large")
        model.eval()
        with torch.no_grad():
            # warm_up
            fake_data=[dataset_dir.encode('utf-8')+b"test2015/re_COCO_test2015_000000262144.jpg"]*32
            model(fake_data)
            torch.cuda.synchronize()
            wait_ready.put(0,block=False)

            while True:    
                ids,datas=bs2ve.get(block=True)

                # datas=model.forward_time(datas)

                # start_time=time.perf_counter()
                datas=model(datas)
                
                # add cache into ids and datas
                output_queue.put([ids,datas],block=False) 

                torch.cuda.synchronize()
                v_time.put([ids,time.perf_counter()],block=False) 
                
                # print(f"batch size of blip_vqa_visual_encoder={len(datas)}")
                
    except KeyboardInterrupt:
        output_queue.close()
        output_queue.cancel_join_thread()
        print("blip_vqa_visual_encoder_engine exit..................................................")
        return 

if __name__ == "__main__":
    try:

        # Create a list to hold process objects
        processes = []

        input_queue=multiprocessing.Queue()
        start_time_queue=multiprocessing.Queue()
        wait_ready=multiprocessing.Queue()
        wait_ready.put(1,block=False)
        input_process=multiprocessing.Process(target=generate_input, args=(input_queue,start_time_queue,wait_ready))
        processes.append(input_process)

        cm2vc=multiprocessing.Queue()
        vc2cm=multiprocessing.Queue()
        cm2c_list=[cm2vc]
        c2cm_list=[vc2cm]
        cm2bs=multiprocessing.Queue()
        cache_manager_process=multiprocessing.Process(target=cache_manager,args=(input_queue,cm2c_list,c2cm_list,cm2bs,))
        processes.append(cache_manager_process)

        vc2ve=multiprocessing.Queue()
        ve2vc=multiprocessing.Queue()
        # blip_vqa_visual_encoder_cache=multiprocessing.Process(target=cache_get_put,args=(cm2vc,vc2cm,vc2ve,ve2vc,))
        # processes.append(blip_vqa_visual_encoder_cache)

        bs2ve=multiprocessing.Queue()
        bs2e_list=[bs2ve]
        bs_policy={"kind":"fix","bses":"4,4,4"}
        # bs_policy={"kind":"random"}
        # bs_policy={"kind":"explict","bses":["2,2,2","4,4,4","8,8,8","16,16,16","32,32,32"],"time_slot":"1"}
        bs_v=multiprocessing.Queue()
        bs_time=[bs_v]
        bach_scheduler_process=multiprocessing.Process(target=bach_scheduler,args=(cm2bs,bs2e_list,bs_policy,bs_time))
        processes.append(bach_scheduler_process)

        output_queue=multiprocessing.Queue()
        v_time=multiprocessing.Queue()
        blip_vqa_visual_encoder_engine_process=multiprocessing.Process(target=blip_vqa_visual_encoder_engine,args=(vc2ve,ve2vc,None,bs2ve,output_queue,v_time,wait_ready))
        processes.append(blip_vqa_visual_encoder_engine_process)
        
        output_process=multiprocessing.Process(target=record_output,args=(output_queue,start_time_queue,bs_time,v_time))
        processes.append(output_process)

        for p in processes:
            p.start()

        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.join()
        print("main exit................................")
