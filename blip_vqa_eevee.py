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
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict

root_path="/dynamic_batch/ee/"
dataset_dir = root_path+"datasets/vqa/"

sys.path.append(root_path)
from models.blip.blip_vqa_visual_encoder import blip_vqa_visual_encoder
from models.blip.blip_vqa_text_encoder import blip_vqa_text_encoder
from models.blip.blip_vqa_text_decoder import blip_vqa_text_decoder


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.start_time_cache = OrderedDict()
        self.living_cache = OrderedDict()
        self.end_time_cache = OrderedDict()

    def get(self, key):
        if key in self.living_cache:
            if time.perf_counter()<self.end_time_cache[key]:
                self.cache.move_to_end(key,last=True)
                self.start_time_cache[key]=time.perf_counter()
                self.start_time_cache.move_to_end(key,last=True)
                self.living_cache.move_to_end(key,last=True)
                self.end_time_cache[key]=self.start_time_cache[key]+self.living_cache[key]
                self.end_time_cache.move_to_end(key,last=True)
                return self.living_cache[key]  
            # in but timeout   
            del self.start_time_cache[key]
            del self.living_cache[key]
            del self.end_time_cache[key]
        return None

    def put(self, key,living):
        if living==0:
            return
        
        if key not in self.living_cache and len(self.living_cache) >= self.capacity:
            self.start_time_cache.popitem(last=False)
            self.living_cache.popitem(last=False)
            self.end_time_cache.popitem(last=False)

        self.start_time_cache[key]=time.perf_counter()
        self.start_time_cache.move_to_end(key,last=True)
        self.living_cache[key]=living
        self.living_cache.move_to_end(key,last=True)
        self.end_time_cache[key]=self.start_time_cache[key]+living
        self.end_time_cache.move_to_end(key,last=True)
    
    def size(self):
        return len(self.living_cache)


class blip_vqa_request:
    def __init__(self,id,datas,livings,priority):
        self.id=id

        self.start_time=time.perf_counter()

        self.priority=priority
        # 0

        self.datas=datas
        # image text

        self.livings=livings
        # image cache

    def set_time(self,time):
        self.start_time=time

def generate_input(input_queue,start_time_queue,wait_ready):
    try:
        time_slot=1

        json_file = dataset_dir+"vqa_test.json"

        with open(json_file) as f:
            dataset = json.load(f)

        # request_num_list=[4,4,4,4]
        request_num_list=[2,4,8,16,32]

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
                tmp.append(blip_vqa_request(id,[image,text],[0],0))
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
        
def record_output(out_queue,start_time_queue,bs_time,e_time):
    try:
        request_num=start_time_queue.get(block=True)
        
        dict_list=[]

        # start_time
        start_time_dict={}
        while len(start_time_dict)<request_num:
            start_time_dict.update(start_time_queue.get(block=True))
        dict_list.append(start_time_dict)
        
        # bs_time
        bs_time_dict={}
        while len(bs_time_dict)<request_num:
            b_t=bs_time[0].get(block=True)
            ids,end_time=b_t[0],b_t[1]
            bs_time_dict.update(dict(zip(ids,[end_time]*len(ids))))
        dict_list.append(bs_time_dict)

        for e in e_time:
            e_time_dict={}
            while len(e_time_dict)<request_num:
                v_t=e.get(block=True)
                ids,end_time=v_t[0],v_t[1]
                e_time_dict.update(dict(zip(ids,[end_time]*len(ids))))
            dict_list.append(e_time_dict)
        
        colors = ['red', 'green', 'blue', 'cyan', 'magenta','yellow','black','white','orange','purple','pink','brown',]
        labels = ['sche','visual','text encoder','text decoder']
        set_label=False
        for r in range(request_num):
            if not set_label:
                set_label=True
                for d in range(len(dict_list)-1):
                    plt.plot([dict_list[d][r],dict_list[d+1][r]],[r]*2,color=colors[d],label=labels[d])
            else:
                for d in range(len(dict_list)-1):
                    plt.plot([dict_list[d][r],dict_list[d+1][r]],[r]*2,color=colors[d])
        plt.xlabel('time')
        plt.ylabel('req id')
        plt.title("time")
        plt.grid(True)
        plt.legend()
        x_min, x_max = min(min(d.values()) for d in dict_list), max(max(d.values()) for d in dict_list)
        plt.xticks(np.arange(x_min, x_max + 1, 1))
        plt.savefig('blip_vqa_eevee.png') 

        print("record_output finish..................................................")
    except KeyboardInterrupt:
        print("record_output exit..................................................")

def cache_manager(input_queue,cm2c_list,c2cm,cm2bs,):
    try:
        for i in range(len(cm2c_list)):
            cm2c_list[i].put(i,block=False)
            # set cache id

        cache_list=[LRUCache(100) for _ in range(len(cm2c_list))]
        # cached

        id2keyliving_list=[{} for _ in range(len(cm2c_list))]
        # ready to cache

        while True:
            try:
                r_b=input_queue.get(block=False)

                cache_id=[]
                cache_in=[]
                for r in r_b:
                    for i in range(len(r.livings)):
                        # every req is ready to cache (again)
                        id2keyliving_list[i][r.id]=[r.datas[i],r.livings[i]]

                        # mark req that is cached
                        living=cache_list[i].get(r.datas[i])
                        if living is not None:
                            r.datas[i]=None
                        else:
                            cache_id.append(r.id)
                            cache_in.append(r.datas[i])
                
                cm2bs.put(r_b,block=False)
                
                # ask cache to send data to engine
                cm2c_list[i].put([cache_id,cache_in],block=False)

                # how to delete data in cache engine????????????????????
                    
            except queue.Empty:
                pass
            try:
                cache_id,ids=c2cm.get(block=False)
                for i in ids:
                    key=id2keyliving_list[cache_id][i][0]
                    living=id2keyliving_list[cache_id][i][1]
                    cache_list[cache_id].put(key,living)
                    del id2keyliving_list[cache_id][i]
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        print("cache_manager exit..................................................")

def cache_engine(cm2c,c2cm,c2e,e2c):
    try:
        cache_id=cm2c.get(block=True)
        cache={}

        while True:
            try:
                ids,in_data,out_data=e2c.get(block=False)

                for i in range(len(ids)):
                    print("a",in_data[i])
                    cache[in_data[i]]=out_data[i:i+1]

                c2cm.put([cache_id,ids],block=False)
                # inform cache manager to add cache info
            except queue.Empty:
                pass
            try:
                ids,cache_in=cm2c.get(block=False)

                cache_out=[]
                for i in cache_in:
                    print("b",i)
                    cache_out.append(cache[i])
                
                c2e.put([ids,cache_out],block=False)
            except queue.Empty:
                pass           
    except KeyboardInterrupt:
        print(f"cache_{cache_id} exit..................................................")
    
def bach_scheduler(cm2bs,bs2e_list,bs_policy,bs_time):
    try:
        # algo need what data struct?
        # append req directly for now
        if bs_policy["kind"]=="fix":
            ids_list=[[] for _ in range(3)]
            datas_list=[[] for _ in range(3)]
            livings_list=[[] for _ in range(3)]
            bs_list=[int(bs) for bs in bs_policy["bses"].split(",")]
            # print(f"bs_list={bs_list}")
            while True:    
                try:
                    r_b=cm2bs.get(block=False)
                    for r in r_b:
                        if r.datas[0] is not None:
                            ids_list[0].append(r.id)
                            datas_list[0].append(r.datas[0])
                            livings_list[0].append(r.livings[0])
                        if r.datas[1] is not None:
                            ids_list[1].append(r.id)
                            datas_list[1].append(r.datas[1])
                        ids_list[2].append(r.id)
                except queue.Empty:
                    pass
                if len(ids_list[0])!=0:
                    # print(f"len(data_list0)={len(data_list0)}")
                    i=0
                    while i+bs_list[0]<=len(ids_list[0]):
                        bs2e_list[0].put([ids_list[0][i:i+bs_list[0]],datas_list[0][i:i+bs_list[0]],livings_list[0][i:i+bs_list[0]]],block=False)
                        bs_time[0].put([ids_list[0][i:i+bs_list[0]],time.perf_counter()],block=False)
                        i+=bs_list[0]
                    if i<len(ids_list[0]):
                        bs2e_list[0].put([ids_list[0][i:],datas_list[0][i:],livings_list[0][i:]],block=False)
                        bs_time[0].put([ids_list[0][i:],time.perf_counter()],block=False)
                    ids_list[0]=[]
                    datas_list[0]=[]
                    livings_list[0]=[]
                if len(ids_list[1])!=0:
                    i=0
                    while i+bs_list[1]<=len(ids_list[1]):
                        bs2e_list[1].put([ids_list[1][i:i+bs_list[1]],datas_list[1][i:i+bs_list[1]]],block=False)
                        bs_time[1].put([ids_list[1][i:i+bs_list[1]],time.perf_counter()],block=False)
                        i+=bs_list[1]
                    if i<len(ids_list[1]):
                        bs2e_list[1].put([ids_list[1][i:],datas_list[1][i:]],block=False)
                        bs_time[1].put([ids_list[1][i:],time.perf_counter()],block=False)
                    ids_list[1]=[]
                    datas_list[1]=[]
                if len(ids_list[2])!=0:
                    i=0
                    while i+bs_list[2]<=len(ids_list[2]):
                        bs2e_list[2].put([ids_list[2][i:i+bs_list[2]],None],block=False)
                        bs_time[2].put([ids_list[2][i:i+bs_list[2]],time.perf_counter()],block=False)
                        i+=bs_list[2]
                    if i<len(ids_list[2]):
                        bs2e_list[2].put([ids_list[2][i:],None],block=False)
                        bs_time[2].put([ids_list[2][i:],time.perf_counter()],block=False)
                    ids_list[2]=[]


    except KeyboardInterrupt:
        print("bach_scheduler exit..................................................")

def blip_vqa_visual_encoder_engine(c2e,e2c,pre_queue,bs2e,output_queue,e_time,wait_ready):
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
                try:
                    ids,in_data,_=bs2e.get(block=True)

                    # start_time=time.perf_counter()
                    out_data=model(in_data)
                    
                    # add cache
                    e2c.put([ids,in_data,out_data],block=False)


                    # add cache into ids and datas
                    # cached_ids,cached_data=c2e.get(block=True)
                    # how big????
                    # cache.get()????

                    output_queue.put([ids,out_data],block=False) 

                    torch.cuda.synchronize()
                    e_time.put([ids,time.perf_counter()],block=False) 
                    
                    # print(f"batch size of blip_vqa_visual_encoder={len(datas)}")
                except queue.Empty:
                    pass   
                try:
                    ids,out_data=c2e.get(block=False)
                    output_queue.put([ids,out_data],block=False) 
                    e_time.put([ids,time.perf_counter()],block=False) 
                except queue.Empty:
                    pass        
    except KeyboardInterrupt:
        output_queue.close()
        output_queue.cancel_join_thread()
        print("blip_vqa_visual_encoder_engine exit..................................................")
        return 

def get_pre_data(now_left,len2send):
    i=0
    data2send=[]
    while i<len2send:
        if len2send-i<len(now_left[0]):
            data2send.append(now_left[0][:len2send-i])
            now_left[0]=now_left[0][len2send-i:]
        else:
            data2send.append(now_left[0])
            now_left=now_left[1:]
        i+=len(data2send[-1])
    return torch.cat(data2send,dim=0)
    
def blip_vqa_text_encoder_engine(c2e,e2c,pre_queue,bs2e,output_queue,e_time,wait_ready):
    try:
        model_url = root_path+"pretrained/model_base_vqa_capfilt_large.pth"
        model = blip_vqa_text_encoder(pretrained=model_url, vit="large")
        model.eval()
        with torch.no_grad():
            # warm_up
            images_embeds = torch.load(root_path+"pretrained/images_embeds.pth")
            images_embeds=images_embeds.repeat(images_embeds.shape[0]*32,*tuple([1]*len(images_embeds.shape[1:])))
            questions = [b"where is the woman sitting?"]*32
            model(images_embeds, questions)
            torch.cuda.synchronize()
            wait_ready.put(0,block=False)

            now_left=[]
            lbs=0
            while True:    
                ids,questions=bs2e.get(block=True)
            
                while lbs<len(questions):
                    now_left.append(pre_queue.get(block=True)[1])
                    lbs+=len(now_left[-1])

                images_embeds=get_pre_data(now_left,len(questions))
                lbs-=len(questions)

                # start_time=time.perf_counter()
                datas=model(images_embeds, questions)
                
                # add cache into ids and datas
                output_queue.put([ids,datas],block=False) 

                torch.cuda.synchronize()
                e_time.put([ids,time.perf_counter()],block=False) 
                
                # print(f"batch size of blip_vqa_visual_encoder={len(datas)}")
                
    except KeyboardInterrupt:
        output_queue.close()
        output_queue.cancel_join_thread()
        print("blip_vqa_text_encoder_engine exit..................................................")
        return 

def blip_vqa_text_decoder_engine(c2e,e2c,pre_queue,bs2e,output_queue,e_time,wait_ready):
    try:
        model_url = root_path+"pretrained/model_base_vqa_capfilt_large.pth"
        model = blip_vqa_text_decoder(pretrained=model_url, vit="large")
        model.eval()
        with torch.no_grad():
            # warm_up
            questions_states=torch.load(root_path+"pretrained/questions_states.pth")
            questions_states=questions_states.repeat(questions_states.shape[0]*32,*tuple([1]*len(questions_states.shape[1:])))

            model(questions_states)
            torch.cuda.synchronize()
            wait_ready.put(0,block=False)

            now_left=[]
            lbs=0
            while True:    
                ids,_=bs2e.get(block=True)
                
                while lbs<len(ids):
                    now_left.append(pre_queue.get(block=True)[1])
                    lbs+=len(now_left[-1])

                questions_states=get_pre_data(now_left,len(ids))
                lbs-=len(ids)

                # start_time=time.perf_counter()
                datas=model(questions_states)
                
                # add cache into ids and datas
                output_queue.put([ids,datas],block=False) 

                torch.cuda.synchronize()
                e_time.put([ids,time.perf_counter()],block=False) 

                # print(f"batch size of blip_vqa_visual_encoder={len(datas)}")
                
    except KeyboardInterrupt:
        output_queue.close()
        output_queue.cancel_join_thread()
        print("blip_vqa_text_encoder_engine exit..................................................")
        return 

if __name__ == "__main__":
    try:

        # Create a list to hold process objects
        processes = []

        input_queue=multiprocessing.Queue()
        start_time_queue=multiprocessing.Queue()
        wait_ready=multiprocessing.Queue()
        wait_ready.put(3,block=False)
        input_process=multiprocessing.Process(target=generate_input, args=(input_queue,start_time_queue,wait_ready))
        processes.append(input_process)

        c2cm=multiprocessing.Queue()
        cm2vc=multiprocessing.Queue()
        cm2c_list=[cm2vc]
        cm2bs=multiprocessing.Queue()
        cache_manager_process=multiprocessing.Process(target=cache_manager,args=(input_queue,cm2c_list,c2cm,cm2bs,))
        processes.append(cache_manager_process)

        vc2ve=multiprocessing.Queue()
        ve2vc=multiprocessing.Queue()
        blip_vqa_visual_encoder_cache_process=multiprocessing.Process(target=cache_engine,args=(cm2vc,c2cm,vc2ve,ve2vc,))
        processes.append(blip_vqa_visual_encoder_cache_process)

        bs2ve=multiprocessing.Queue()
        bs2tee=multiprocessing.Queue()
        bs2tde=multiprocessing.Queue()
        bs2e_list=[bs2ve,bs2tee,bs2tde]
        bs_v=multiprocessing.Queue()
        bs_te=multiprocessing.Queue()
        bs_td=multiprocessing.Queue()
        bs_time=[bs_v,bs_te,bs_td]
        bs_policy={"kind":"fix","bses":"4,4,4"}#0.913968265867762
        #bs_policy={"kind":"fix","bses":"2,4,8"}#1.0067323830669686
        # bs_policy={"kind":"random"}
        # bs_policy={"kind":"explict","bses":["2,2,2","4,4,4","8,8,8","16,16,16","32,32,32"],"time_slot":"1"}
        bach_scheduler_process=multiprocessing.Process(target=bach_scheduler,args=(cm2bs,bs2e_list,bs_policy,bs_time))
        processes.append(bach_scheduler_process)

        v2te=multiprocessing.Queue()
        v_time=multiprocessing.Queue()
        blip_vqa_visual_encoder_engine_process=multiprocessing.Process(target=blip_vqa_visual_encoder_engine,args=(vc2ve,ve2vc,None,bs2ve,v2te,v_time,wait_ready))
        processes.append(blip_vqa_visual_encoder_engine_process)
        
        te2td=multiprocessing.Queue()
        te_time=multiprocessing.Queue()
        blip_vqa_text_encoder_engine_process=multiprocessing.Process(target=blip_vqa_text_encoder_engine,args=(None,None,v2te,bs2tee,te2td,te_time,wait_ready))
        processes.append(blip_vqa_text_encoder_engine_process)

        output_queue=multiprocessing.Queue()
        td_time=multiprocessing.Queue()
        blip_vqa_text_decoder_engine_process=multiprocessing.Process(target=blip_vqa_text_decoder_engine,args=(None,None,te2td,bs2tde,output_queue,td_time,wait_ready))
        processes.append(blip_vqa_text_decoder_engine_process)

        e_time=[v_time,te_time,td_time]
        output_process=multiprocessing.Process(target=record_output,args=(output_queue,start_time_queue,bs_time,e_time))
        processes.append(output_process)

        for p in processes:
            p.start()

        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.join()
        print("main exit................................")
