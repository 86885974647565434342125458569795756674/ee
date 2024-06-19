import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import tritonclient.http as httpclient
from tritonclient.utils import *
import time
import multiprocessing
import queue
import random
import os
from cache import cache_get_put,recover_cache_duplication,recover_once
from utils import while_client

def blip_vqa_visual_encoder_task(input_queue,output_queue):
    try:
        processes=[]
        
        blip_vqa_visual_encoder_input_queue=multiprocessing.Queue()
        blip_vqa_visual_encoder_output_queue=multiprocessing.Queue()
        blip_vqa_visual_encoder_process = multiprocessing.Process(target=while_client,args=("blip_vqa_visual_encoder",blip_vqa_visual_encoder_input_queue,blip_vqa_visual_encoder_output_queue))
        processes.append(blip_vqa_visual_encoder_process)
        blip_vqa_visual_encoder_process.start()

        cache_get_input_queue=multiprocessing.Queue()
        cache_get_output_queue=multiprocessing.Queue()
        cache_put_queue=multiprocessing.Queue()
        cache_process=multiprocessing.Process(target=cache_get_put,args=(cache_get_input_queue,cache_get_output_queue,cache_put_queue,))
        processes.append(cache_process)
        cache_process.start()

        while True:
            images,blip_vqa_visual_encoder_batch_size,livings=input_queue.get()
            cache_get_input_queue.put((images,livings),block=False)
            unique_images,cached_images_result,no_forward_indices,unique_images_forward,forward_indices,livings_forward=cache_get_output_queue.get()
            
            if unique_images_forward.shape[0]==0:
                blip_vqa_visual_encoder_batches=[]                    
                forward_indices_batches=[]
                livings_batches=[]
                num_blip_vqa_visual_encoder_batch=0
            elif unique_images_forward.shape[0]<=blip_vqa_visual_encoder_batch_size:
                blip_vqa_visual_encoder_batches=[unique_images_forward]               
                forward_indices_batches=[forward_indices]
                livings_batches=[livings_forward]     
                num_blip_vqa_visual_encoder_batch=1
            else:
                num_blip_vqa_visual_encoder_batch=unique_images_forward.shape[0]//blip_vqa_visual_encoder_batch_size
                num_blip_vqa_visual_encoder_left=unique_images_forward.shape[0]-num_blip_vqa_visual_encoder_batch*blip_vqa_visual_encoder_batch_size
                if num_blip_vqa_visual_encoder_left!=0:
                    blip_vqa_visual_encoder_batches=np.split(unique_images_forward[:-num_blip_vqa_visual_encoder_left],num_blip_vqa_visual_encoder_batch)+[unique_images_forward[-num_blip_vqa_visual_encoder_left:]]
                    livings_batches=np.split(livings_forward[:-num_blip_vqa_visual_encoder_left],num_blip_vqa_visual_encoder_batch)+[livings_forward[-num_blip_vqa_visual_encoder_left:]]
                    forward_indices_batches=np.split(forward_indices[:-num_blip_vqa_visual_encoder_left],num_blip_vqa_visual_encoder_batch)+[forward_indices[-num_blip_vqa_visual_encoder_left:]]
                    num_blip_vqa_visual_encoder_batch+=1
                else:
                    blip_vqa_visual_encoder_batches=np.split(unique_images_forward,num_blip_vqa_visual_encoder_batch)
                    livings_batches=np.split(livings_forward,num_blip_vqa_visual_encoder_batch)
                    forward_indices_batches=np.split(forward_indices,num_blip_vqa_visual_encoder_batch)

            if len(blip_vqa_visual_encoder_batches)==0:
                # all cached
                recover_images_from_duplication=recover_once(images,unique_images,cached_images_result)
                output_queue.put(recover_images_from_duplication)
                continue

            for images_per_batch in blip_vqa_visual_encoder_batches:
                blip_vqa_visual_encoder_input_queue.put(images_per_batch,block=False)

            recover_images_from_duplication=None
            mask=None
            no_forward_indices_indice_start=0
            start_to_send_indice=0

            for i,images_per_batch in enumerate(blip_vqa_visual_encoder_batches):

                output_per_batch=blip_vqa_visual_encoder_output_queue.get()

                # add cache
                cache_put_queue.put((images_per_batch,output_per_batch,livings_batches[i]),block=False)
                
                # print("np.all((no_forward_indices>forward_indices_batches[i][-1])==False)")
                # print(np.all((no_forward_indices>forward_indices_batches[i][-1])==False))
                if np.all((no_forward_indices>forward_indices_batches[i][-1])==False):
                    # no_forward_indices all smaller than the last element of forward_indices_batches[i]
                    # all cached can be forward in this time
                    no_forward_indices_indice=no_forward_indices.shape[0]-1       
                else:
                    no_forward_indices_indice=np.argmax(no_forward_indices>forward_indices_batches[i][-1])
                    # find the first element of no_forward_indices bigger than the last element of forward_indices_batches[i]
                    # the last indice of cached can be forward in this time
                if recover_images_from_duplication is None:
                    recover_images_from_duplication=np.empty((images.shape[0],*(output_per_batch.shape[1:])),dtype=output_per_batch.dtype)
                    mask=np.full(images.shape[0],False)

                recover_images_from_duplication,mask=recover_cache_duplication(images,unique_images,cached_images_result[no_forward_indices_indice_start:no_forward_indices_indice+1],no_forward_indices[no_forward_indices_indice_start:no_forward_indices_indice+1],output_per_batch,forward_indices_batches[i],recover_images_from_duplication,mask)
                no_forward_indices_indice_start=no_forward_indices_indice+1
                # print(mask)

                # send as you can
                inc=0
                while start_to_send_indice<images.shape[0] and start_to_send_indice+inc<images.shape[0] and np.all(mask[start_to_send_indice:start_to_send_indice+inc+1] == True):
                    inc+=1
                # print("inc")
                # print(inc)
                output_queue.put(recover_images_from_duplication[start_to_send_indice:start_to_send_indice+inc],block=False)
                start_to_send_indice+=inc

    except KeyboardInterrupt:
        for process in processes:
            process.join()

def blip_vqa_text_encoder_task(input_queue,pre_stage_queue,output_queue):
    try:
        processes=[]
        
        blip_vqa_text_encoder_input_queue=multiprocessing.Queue()
        blip_vqa_text_encoder_output_queue=multiprocessing.Queue()
        blip_vqa_text_encoder_process = multiprocessing.Process(target=while_client,args=("blip_vqa_text_encoder",blip_vqa_text_encoder_input_queue,blip_vqa_text_encoder_output_queue))
        processes.append(blip_vqa_text_encoder_process)
        blip_vqa_text_encoder_process.start()

        while True:
            texts,blip_vqa_text_encoder_batch_size=input_queue.get()

            if texts.shape[0]<=blip_vqa_text_encoder_batch_size:
                blip_vqa_text_encoder_batches=[texts]
                num_blip_vqa_text_encoder_batch=1
            else:
                num_blip_vqa_text_encoder_batch=texts.shape[0]//blip_vqa_text_encoder_batch_size
                num_blip_vqa_text_encoder_left=texts.shape[0]-num_blip_vqa_text_encoder_batch*blip_vqa_text_encoder_batch_size
                if num_blip_vqa_text_encoder_left!=0:
                    blip_vqa_text_encoder_batches=np.split(texts[:-num_blip_vqa_text_encoder_left],num_blip_vqa_text_encoder_batch)+[texts[-num_blip_vqa_text_encoder_left:]]
                    num_blip_vqa_text_encoder_batch+=1
                else:
                    blip_vqa_text_encoder_batches=np.split(texts,num_blip_vqa_text_encoder_batch)

            text_batch_num=len(blip_vqa_text_encoder_batches)
            # print(text_batch_num)
            text_batch_input_count=0
            text_batch_output_count=0
            now_left=None
            while text_batch_input_count<text_batch_num or text_batch_output_count<text_batch_num:
                # cache pad image replication
                try:
                    images_per_batch=pre_stage_queue.get(block=False)
                    if now_left is None:
                        now_left=images_per_batch
                    else:
                        now_left=np.concatenate([now_left,images_per_batch],axis=0)
                    # print(text_batch_input_count,now_left.shape[0],blip_vqa_text_encoder_batches[text_batch_input_count].shape[0])
                except queue.Empty:
                    pass

                while now_left is not None and text_batch_input_count<text_batch_num and now_left.shape[0]>=blip_vqa_text_encoder_batches[text_batch_input_count].shape[0]:
                    # print(blip_vqa_text_encoder_batches[text_batch_input_count])
                    blip_vqa_text_encoder_input_queue.put((now_left[:blip_vqa_text_encoder_batches[text_batch_input_count].shape[0]],blip_vqa_text_encoder_batches[text_batch_input_count]),block=False)
                    now_left=now_left[blip_vqa_text_encoder_batches[text_batch_input_count].shape[0]:]
                    text_batch_input_count+=1

                try:
                    texts_per_batch=blip_vqa_text_encoder_output_queue.get(block=False)
                    text_batch_output_count+=1
                    output_queue.put(texts_per_batch,block=False)
                except queue.Empty:
                    pass      
    except KeyboardInterrupt:
        for process in processes:
            process.join()


def pad_concate(input0_data,input1_data,input0_data1):
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

def blip_vqa_text_decoder_task(input_queue,pre_stage_queue,output_queue):
    try:
        processes=[]

        blip_vqa_text_decoder_input_queue=multiprocessing.Queue()
        blip_vqa_text_decoder_output_queue=multiprocessing.Queue()
        blip_vqa_text_decoder_process = multiprocessing.Process(target=while_client,args=("blip_vqa_text_decoder",blip_vqa_text_decoder_input_queue,blip_vqa_text_decoder_output_queue))
        processes.append(blip_vqa_text_decoder_process)
        blip_vqa_text_decoder_process.start()

        while True:
            texts,blip_vqa_text_decoder_batch_size=input_queue.get()

            if texts.shape[0]<=blip_vqa_text_decoder_batch_size:
                blip_vqa_text_decoder_batches=[texts]
                num_blip_vqa_text_decoder_batch=1
            else:
                num_blip_vqa_text_decoder_batch=texts.shape[0]//blip_vqa_text_decoder_batch_size
                num_blip_vqa_text_decoder_left=texts.shape[0]-num_blip_vqa_text_decoder_batch*blip_vqa_text_decoder_batch_size
                if num_blip_vqa_text_decoder_left!=0:
                    blip_vqa_text_decoder_batches=np.split(texts[:-num_blip_vqa_text_decoder_left],num_blip_vqa_text_decoder_batch)+[texts[-num_blip_vqa_text_decoder_left:]]
                    num_blip_vqa_text_decoder_batch+=1
                else:
                    blip_vqa_text_decoder_batches=np.split(texts,num_blip_vqa_text_decoder_batch)

            text_batch_num=len(blip_vqa_text_decoder_batches)
            text_batch_input_count=0
            text_batch_output_count=0
            now_left_data,now_left_mask=None,None
            while text_batch_input_count<text_batch_num or text_batch_output_count<text_batch_num:
                try:
                    texts_per_batch=pre_stage_queue.get(block=False)
                    if now_left_data is None:
                        now_left_data=texts_per_batch
                        now_left_mask_shape=(now_left_data.shape[0]*now_left_data.shape[1],now_left_data.shape[2])
                        now_left_mask = torch.ones(now_left_mask_shape, dtype=torch.long).numpy(force=True)
                    else:                        
                        now_left_data,now_left_mask=pad_concate(now_left_data,now_left_mask,texts_per_batch)
                    # print(text_batch_input_count,now_left.shape[0],blip_vqa_text_decoder_batches[text_batch_input_count].shape[0])
                except queue.Empty:
                    pass

                while now_left_data is not None and text_batch_input_count<text_batch_num and now_left_data.shape[0]>=blip_vqa_text_decoder_batches[text_batch_input_count].shape[0]:
                    # print(blip_vqa_text_decoder_batches[text_batch_input_count])
                    blip_vqa_text_decoder_input_queue.put((now_left_data[:blip_vqa_text_decoder_batches[text_batch_input_count].shape[0]],now_left_mask[:blip_vqa_text_decoder_batches[text_batch_input_count].shape[0]]),block=False)
                    now_left_data=now_left_data[blip_vqa_text_decoder_batches[text_batch_input_count].shape[0]:]
                    now_left_mask=now_left_mask[blip_vqa_text_decoder_batches[text_batch_input_count].shape[0]:]
                    text_batch_input_count+=1
                
                try:
                    texts_per_batch=blip_vqa_text_decoder_output_queue.get(block=False)
                    text_batch_output_count+=1
                    output_queue.put(texts_per_batch,block=False)
                except queue.Empty:
                    pass       
    except KeyboardInterrupt:
        for process in processes:
            process.join()

def change_batch_size(batch_size_queue,time_interval):
    try:
        while True:
            a,b,c=random.randint(1, 10),random.randint(1, 10),random.randint(1, 10)
            batch_size_queue.put((a,b,c),block=False)
            # print(f"release batch size: {a},{b},{c}")
            time.sleep(time_interval)
    except KeyboardInterrupt:
        pass

def blip_vqa_process(request_queue,request_events,processed_results,batch_size_queue,fix_batch=False):
    try:
        # write_file="/workspace/blip_vqa_process.txt"
        # if(os.path.isfile(write_file)):    
        #     os.remove(write_file)

        processes = []

        blip_vqa_visual_encoder_blip_vqa_text_encoder_queue=multiprocessing.Queue()
        blip_vqa_text_encoder_blip_vqa_text_decoder_queue=multiprocessing.Queue()

        blip_vqa_visual_encoder_batches_queue=multiprocessing.Queue()
        blip_vqa_visual_encoder_task_process=multiprocessing.Process(target=blip_vqa_visual_encoder_task,args=(blip_vqa_visual_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue,))
        processes.append(blip_vqa_visual_encoder_task_process)
        blip_vqa_visual_encoder_task_process.start()

        blip_vqa_text_encoder_batches_queue=multiprocessing.Queue()
        blip_vqa_text_encoder_task_process=multiprocessing.Process(target=blip_vqa_text_encoder_task,args=(blip_vqa_text_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue,blip_vqa_text_encoder_blip_vqa_text_decoder_queue,))
        processes.append(blip_vqa_text_encoder_task_process)
        blip_vqa_text_encoder_task_process.start()

        blip_vqa_text_decoder_batches_queue=multiprocessing.Queue()
        blip_vqa_text_decoder_batches_return_queue=multiprocessing.Queue()
        blip_vqa_text_decoder_task_process=multiprocessing.Process(target=blip_vqa_text_decoder_task,args=(blip_vqa_text_decoder_batches_queue,blip_vqa_text_encoder_blip_vqa_text_decoder_queue,blip_vqa_text_decoder_batches_return_queue,))
        processes.append(blip_vqa_text_decoder_task_process)
        blip_vqa_text_decoder_task_process.start()

        if fix_batch != False:
            blip_vqa_visual_encoder_batch_size,blip_vqa_text_encoder_batch_size,blip_vqa_text_decoder_batch_size=fix_batch
            print(f"fix batch sizes: {blip_vqa_visual_encoder_batch_size},{blip_vqa_text_encoder_batch_size},{blip_vqa_text_decoder_batch_size}")
        
        while True:
            if fix_batch==False:
                try:
                    batch_sizes=batch_size_queue.get(block=False)
                    blip_vqa_visual_encoder_batch_size,blip_vqa_text_encoder_batch_size,blip_vqa_text_decoder_batch_size=batch_sizes
                    print(f"update batch sizes: {batch_sizes}")
                except queue.Empty:
                    pass
            if not request_queue.empty():
                request_ids,batch_nums,images,texts,livings=request_queue.get()
                # print(f"request num: {len(request_ids)}")

                images=np.concatenate(images, axis=0)
                texts=np.concatenate(texts, axis=0)
                livings=np.concatenate(livings, axis=0)

                blip_vqa_visual_encoder_batches_queue.put((images,blip_vqa_visual_encoder_batch_size,livings),block=False)            

                blip_vqa_text_encoder_batches_queue.put((texts,blip_vqa_text_encoder_batch_size),block=False)

                blip_vqa_text_decoder_batches_queue.put((texts,blip_vqa_text_decoder_batch_size),block=False)                

                if texts.shape[0]<=blip_vqa_text_decoder_batch_size:
                    num_blip_vqa_text_decoder_batch=1
                else:
                    num_blip_vqa_text_decoder_batch=texts.shape[0]//blip_vqa_text_decoder_batch_size
                    num_blip_vqa_text_decoder_left=texts.shape[0]-num_blip_vqa_text_decoder_batch*blip_vqa_text_decoder_batch_size
                    if num_blip_vqa_text_decoder_left!=0:
                        num_blip_vqa_text_decoder_batch+=1
                
                batch_count=0
                now_left=None
                for _ in range(num_blip_vqa_text_decoder_batch):
                    blip_vqa_text_decoder_batches_return=blip_vqa_text_decoder_batches_return_queue.get()
                    if now_left is None:
                        now_left=blip_vqa_text_decoder_batches_return
                    else:
                        now_left=np.concatenate([now_left,blip_vqa_text_decoder_batches_return],axis=0)
                    while batch_count<len(batch_nums) and now_left.shape[0]>=batch_nums[batch_count]:
                        post_return=now_left[:batch_nums[batch_count]]
                        now_left=now_left[batch_nums[batch_count]:]
                        processed_results[request_ids[batch_count]] = post_return
                        request_events[request_ids[batch_count]]=1
                        # print(f"one request finish, batch size: {batch_nums[batch_count]}, request id: {batch_count}")
                        batch_count+=1      
    except KeyboardInterrupt:
        for process in processes:
            process.join()

if __name__=="__main__":

    try:

        # Create a list to hold process objects
        processes = []

        batch_size_queue=multiprocessing.Queue()
        # queue is not the fastest way, maybe pipe
        time_interval=1

        change_batch_size_process = multiprocessing.Process(target=change_batch_size, args=(batch_size_queue,time_interval))
        processes.append(change_batch_size_process)
        change_batch_size_process.start()

        blip_vqa_visual_encoder_blip_vqa_text_encoder_queue=multiprocessing.Queue()
        blip_vqa_text_encoder_blip_vqa_text_decoder_queue=multiprocessing.Queue()

        blip_vqa_visual_encoder_batches_queue=multiprocessing.Queue()
        blip_vqa_visual_encoder_task_process=multiprocessing.Process(target=blip_vqa_visual_encoder_task,args=(blip_vqa_visual_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue,))
        processes.append(blip_vqa_visual_encoder_task_process)
        blip_vqa_visual_encoder_task_process.start()

        blip_vqa_visual_encoder_batch_size,blip_vqa_text_encoder_batch_size,blip_vqa_text_decoder_batch_size=1,1,1

        for _ in range(3):
            request_num=random.randint(1, 10)
            request_num=1
            batch_size_list=[random.randint(1,5)*2 for _ in range(request_num)]
            batch_size_list=[10]
            images_batches = [
                np.array([b"/workspace/demos/images/merlion.png",b"/workspace/demos/images/beach.jpg"]*(batch_size//2)) for batch_size in batch_size_list
            ]
            texts_batches = [
                np.array([b"where is it?",b"where is the woman sitting?"]*(batch_size//2)) for batch_size in batch_size_list
            ]
            livings_batches = [
                np.array([random.randint(1, 10),random.randint(1, 10)]*(batch_size//2)) for batch_size in batch_size_list
            ]

            request_ids, batch_nums,images,texts,livings=[],[],[],[],[]
            for i in range(request_num):
                request_id,image_batch,text_batch,living_batch=i,images_batches[i],texts_batches[i],livings_batches[i]
                request_ids.append(request_id)
                images.append(image_batch)
                texts.append(text_batch)
                livings.append(living_batch)
                batch_nums.append(image_batch.shape[0])        

            try:
                batch_sizes=batch_size_queue.get(block=False)
                blip_vqa_visual_encoder_batch_size,blip_vqa_text_encoder_batch_size,blip_vqa_text_decoder_batch_size=batch_sizes[0],batch_sizes[1],batch_sizes[2]
                print(f"update batch sizes: {batch_sizes}")
            except queue.Empty:
                pass

            images=np.concatenate(images, axis=0)
            texts=np.concatenate(texts, axis=0)
            livings=np.concatenate(livings, axis=0)

            blip_vqa_visual_encoder_batches_queue.put((images,livings,blip_vqa_visual_encoder_batch_size),block=False)            

            has_output_num=0
            while True:
                output_per_batch=blip_vqa_visual_encoder_blip_vqa_text_encoder_queue.get()
                print("output_per_batch.shape")
                print(output_per_batch.shape)
                has_output_num+=output_per_batch.shape[0]
                print(f"has_output_num: {has_output_num}")
                if has_output_num==images.shape[0]:
                    break
            print("------------------------------------------------")
        
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.join()
        print("finish...")
