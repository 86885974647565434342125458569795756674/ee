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
        
def record_output(out_queue,start_time_queue,bs_time,e_time):
    try:
        request_num=start_time_queue.get(block=True)

        # # start_time
        start_time_dict={}
        while len(start_time_dict)<request_num:
            start_time_dict.update(start_time_queue.get(block=True))

        dict_list=[]
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
        
        for i in range(request_num):
            s=f"id={i},start_time={start_time_dict[i]},end_time={dict_list[-1][i]},latency={dict_list[-1][i]-start_time_dict[i]},bs_time-start_time={bs_time_dict[i]-start_time_dict[i]}"
            for d in range(1,len(dict_list)):
                s=s+f",{d}_e-{d-1}_e={dict_list[d][i]-dict_list[d-1][i]}"
            print(s)
        
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
            data_list0=[]
            data_ids0=[]
            data_list1=[]
            data_ids1=[]
            data_ids2=[]
            bs_list=[]
            bs_list=bs_list+[int(bs) for bs in bs_policy["bses"].split(",")]
            # print(f"bs_list={bs_list}")
            while True:    
                try:
                    r_b=cm2bs.get(block=False)
                    for r in r_b:
                        if r.image is not None:
                            data_list0.append(r.image)
                            data_ids0.append(r.id)
                        if r.text is not None:
                            data_list1.append(r.text)
                            data_ids1.append(r.id)
                        data_ids2.append(r.id)
                except queue.Empty:
                    pass
                if len(data_ids0)!=0:
                    # print(f"len(data_list0)={len(data_list0)}")
                    i=0
                    while i+bs_list[0]<=len(data_ids0):
                        bs2e_list[0].put([data_ids0[i:i+bs_list[0]],data_list0[i:i+bs_list[0]]],block=False)
                        bs_time[0].put([data_ids0[i:i+bs_list[0]],time.perf_counter()],block=False)
                        i+=bs_list[0]
                    if i<len(data_ids0):
                        bs2e_list[0].put([data_ids0[i:],data_list0[i:]],block=False)
                        bs_time[0].put([data_ids0[i:],time.perf_counter()],block=False)
                    data_list0=[]
                    data_ids0=[]
                if len(data_ids1)!=0:
                    i=0
                    while i+bs_list[1]<=len(data_ids1):
                        bs2e_list[1].put([data_ids1[i:i+bs_list[1]],data_list1[i:i+bs_list[1]]],block=False)
                        bs_time[1].put([data_ids1[i:i+bs_list[1]],time.perf_counter()],block=False)
                        i+=bs_list[1]
                    if i<len(data_ids1):
                        bs2e_list[1].put([data_ids1[i:],data_list1[i:]],block=False)
                        bs_time[1].put([data_ids1[i:],time.perf_counter()],block=False)
                    data_list1=[]
                    data_ids1=[]
                if len(data_ids2)!=0:
                    i=0
                    while i+bs_list[2]<=len(data_ids2):
                        bs2e_list[2].put([data_ids2[i:i+bs_list[2]],None],block=False)
                        bs_time[2].put([data_ids2[i:i+bs_list[2]],time.perf_counter()],block=False)
                        i+=bs_list[2]
                    if i<len(data_ids2):
                        bs2e_list[2].put([data_ids2[i:],None],block=False)
                        bs_time[2].put([data_ids2[i:],time.perf_counter()],block=False)
                    data_ids2=[]


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
                ids,datas=bs2e.get(block=True)

                # datas=model.forward_time(datas)

                # start_time=time.perf_counter()
                datas=model(datas)
                
                # add cache into ids and datas
                output_queue.put([ids,datas],block=False) 

                torch.cuda.synchronize()
                e_time.put([ids,time.perf_counter()],block=False) 
                
                # print(f"batch size of blip_vqa_visual_encoder={len(datas)}")
                
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


def pad_concate(now_left,len2send):
    input1_data1_shape=(input0_data1.shape[0]*input0_data1.shape[1],input0_data1.shape[2])
    input1_data1 = torch.ones(input1_data1_shape, dtype=torch.long).numpy(force=True)

    if input0_data.shape[2]>input0_data1.shape[2]:
        input1_data1=np.pad(input1_data1,((0,0),(input0_data.shape[2]-input0_data1.shape[2],0)),"constant", constant_values=(0, 0))
        input0_data1=np.pad(input0_data1,((0,0),(0,0),(input0_data.shape[2]-input0_data1.shape[2],0),(0,0)),"constant")
    elif input0_data.shape[2]<input0_data1.shape[2]:
        input1_data=np.pad(input1_data,((0,0),(input0_data1.shape[2]-input0_data.shape[2],0)),"constant", constant_values=(0, 0))
        input0_data=np.pad(input0_data,((0,0),(0,0),(input0_data1.shape[2]-input0_data.shape[2],0),(0,0)),"constant")

    input0_data=np.concatenate([input0_data,input0_data1],axis=0)
    input1_data=np.concatenate([input1_data,input1_data1],axis=0)
    return input0_data,input1_data

   
def blip_vqa_text_decoder_engine(c2e,e2c,pre_queue,bs2e,output_queue,e_time,wait_ready):
    try:
        model_url = root_path+"pretrained/model_base_vqa_capfilt_large.pth"
        model = blip_vqa_text_decoder(pretrained=model_url, vit="large")
        model.eval()
        with torch.no_grad():
            # warm_up
            questions_states=torch.load(root_path+"pretrained/questions_states.pth")
            questions_states=questions_states.repeat(questions_states.shape[0]*32,*tuple([1]*len(questions_states.shape[1:])))

            questions_atts_shape=(questions_states.shape[0]*questions_states.shape[1],questions_states.shape[2])
            questions_atts = torch.ones(questions_atts_shape, dtype=torch.long)
            model(questions_states,questions_atts)
            torch.cuda.synchronize()
            wait_ready.put(0,block=False)

            now_left=[]
            lbs=0
            while True:    
                ids,_=bs2e.get(block=True)
            
                while lbs<len(ids):
                    now_left.append(pre_queue.get(block=True)[1])
                    lbs+=len(now_left[-1])

                questions_states,questions_atts=pad_concate(now_left,len(ids))
                lbs-=len(ids)

                # start_time=time.perf_counter()
                datas=model(questions_states, questions_atts)
                
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
        bs2tee=multiprocessing.Queue()
        bs2e_list=[bs2ve,bs2tee]
        bs_v=multiprocessing.Queue()
        bs_te=multiprocessing.Queue()
        bs_time=[bs_v,bs_te]
        bs_policy={"kind":"fix","bses":"4,4,4"}
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
        blip_vqa_text_decoder_engine_process=multiprocessing.Process(target=blip_vqa_text_decoder_engine,args=(None,None,te2td,bs2tee,output_queue,td_time,wait_ready))
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
