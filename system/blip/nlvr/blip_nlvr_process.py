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

def blip_nlvr_visual_encoder_task(input_queue,output_queue):
    try:
        processes=[]
        
        blip_nlvr_visual_encoder_input_queue=multiprocessing.Queue()
        blip_nlvr_visual_encoder_output_queue=multiprocessing.Queue()
        blip_nlvr_visual_encoder_process = multiprocessing.Process(target=while_client,args=("blip_nlvr_visual_encoder",blip_nlvr_visual_encoder_input_queue,blip_nlvr_visual_encoder_output_queue))
        processes.append(blip_nlvr_visual_encoder_process)
        blip_nlvr_visual_encoder_process.start()

        cache_get_input_queue=multiprocessing.Queue()
        cache_get_output_queue=multiprocessing.Queue()
        cache_put_queue=multiprocessing.Queue()
        cache_process=multiprocessing.Process(target=cache_get_put,args=(cache_get_input_queue,cache_get_output_queue,cache_put_queue,))
        processes.append(cache_process)
        cache_process.start()

        while True:
            images0,image1s,blip_nlvr_visual_encoder_batch_size,livings=input_queue.get()
            cache_get_input_queue.put((images,livings),block=False)
            unique_images,cached_images_result,no_forward_indices,unique_images_forward,forward_indices,livings_forward=cache_get_output_queue.get()
            
            if unique_images_forward.shape[0]==0:
                blip_nlvr_visual_encoder_batches=[]                    
                forward_indices_batches=[]
                livings_batches=[]
                num_blip_nlvr_visual_encoder_batch=0
            elif unique_images_forward.shape[0]<=blip_nlvr_visual_encoder_batch_size:
                blip_nlvr_visual_encoder_batches=[unique_images_forward]               
                forward_indices_batches=[forward_indices]
                livings_batches=[livings_forward]     
                num_blip_nlvr_visual_encoder_batch=1
            else:
                num_blip_nlvr_visual_encoder_batch=unique_images_forward.shape[0]//blip_nlvr_visual_encoder_batch_size
                num_blip_nlvr_visual_encoder_left=unique_images_forward.shape[0]-num_blip_nlvr_visual_encoder_batch*blip_nlvr_visual_encoder_batch_size
                if num_blip_nlvr_visual_encoder_left!=0:
                    blip_nlvr_visual_encoder_batches=np.split(unique_images_forward[:-num_blip_nlvr_visual_encoder_left],num_blip_nlvr_visual_encoder_batch)+[unique_images_forward[-num_blip_nlvr_visual_encoder_left:]]
                    livings_batches=np.split(livings_forward[:-num_blip_nlvr_visual_encoder_left],num_blip_nlvr_visual_encoder_batch)+[livings_forward[-num_blip_nlvr_visual_encoder_left:]]
                    forward_indices_batches=np.split(forward_indices[:-num_blip_nlvr_visual_encoder_left],num_blip_nlvr_visual_encoder_batch)+[forward_indices[-num_blip_nlvr_visual_encoder_left:]]
                    num_blip_nlvr_visual_encoder_batch+=1
                else:
                    blip_nlvr_visual_encoder_batches=np.split(unique_images_forward,num_blip_nlvr_visual_encoder_batch)
                    livings_batches=np.split(livings_forward,num_blip_nlvr_visual_encoder_batch)
                    forward_indices_batches=np.split(forward_indices,num_blip_nlvr_visual_encoder_batch)

            if len(blip_nlvr_visual_encoder_batches)==0:
                # all cached
                recover_images_from_duplication=recover_once(images,unique_images,cached_images_result)
                output_queue.put(recover_images_from_duplication)
                continue

            for images_per_batch in blip_nlvr_visual_encoder_batches:
                blip_nlvr_visual_encoder_input_queue.put(images_per_batch,block=False)

            recover_images_from_duplication=None
            mask=None
            no_forward_indices_indice_start=0
            start_to_send_indice=0

            for i,images_per_batch in enumerate(blip_nlvr_visual_encoder_batches):

                output_per_batch=blip_nlvr_visual_encoder_output_queue.get()

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

def blip_nlvr_text_encoder_task(input_queue,pre_stage_queue,output_queue):
    try:
        processes=[]
        
        blip_nlvr_text_encoder_input_queue=multiprocessing.Queue()
        blip_nlvr_text_encoder_output_queue=multiprocessing.Queue()
        blip_nlvr_text_encoder_process = multiprocessing.Process(target=while_client,args=("blip_nlvr_text_encoder",blip_nlvr_text_encoder_input_queue,blip_nlvr_text_encoder_output_queue))
        processes.append(blip_nlvr_text_encoder_process)
        blip_nlvr_text_encoder_process.start()

        while True:
            texts,blip_nlvr_text_encoder_batch_size=input_queue.get()

            if texts.shape[0]<=blip_nlvr_text_encoder_batch_size:
                blip_nlvr_text_encoder_batches=[texts]
                num_blip_nlvr_text_encoder_batch=1
            else:
                num_blip_nlvr_text_encoder_batch=texts.shape[0]//blip_nlvr_text_encoder_batch_size
                num_blip_nlvr_text_encoder_left=texts.shape[0]-num_blip_nlvr_text_encoder_batch*blip_nlvr_text_encoder_batch_size
                if num_blip_nlvr_text_encoder_left!=0:
                    blip_nlvr_text_encoder_batches=np.split(texts[:-num_blip_nlvr_text_encoder_left],num_blip_nlvr_text_encoder_batch)+[texts[-num_blip_nlvr_text_encoder_left:]]
                    num_blip_nlvr_text_encoder_batch+=1
                else:
                    blip_nlvr_text_encoder_batches=np.split(texts,num_blip_nlvr_text_encoder_batch)

            text_batch_num=len(blip_nlvr_text_encoder_batches)
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
                    # print(text_batch_input_count,now_left.shape[0],blip_nlvr_text_encoder_batches[text_batch_input_count].shape[0])
                except queue.Empty:
                    pass

                while now_left is not None and text_batch_input_count<text_batch_num and now_left.shape[0]>=blip_nlvr_text_encoder_batches[text_batch_input_count].shape[0]:
                    # print(blip_nlvr_text_encoder_batches[text_batch_input_count])
                    blip_nlvr_text_encoder_input_queue.put((now_left[:blip_nlvr_text_encoder_batches[text_batch_input_count].shape[0]],blip_nlvr_text_encoder_batches[text_batch_input_count]),block=False)
                    now_left=now_left[blip_nlvr_text_encoder_batches[text_batch_input_count].shape[0]:]
                    text_batch_input_count+=1

                try:
                    texts_per_batch=blip_nlvr_text_encoder_output_queue.get(block=False)
                    text_batch_output_count+=1
                    output_queue.put(texts_per_batch,block=False)
                except queue.Empty:
                    pass      
    except KeyboardInterrupt:
        for process in processes:
            process.join()


def blip_nlvr_process(request_queue,request_events,processed_results,batch_size_queue,fix_batch=False):
    try:
        # write_file="/workspace/blip_nlvr_process.txt"
        # if(os.path.isfile(write_file)):    
        #     os.remove(write_file)

        processes = []

        blip_nlvr_visual_encoder_blip_nlvr_text_encoder_queue=multiprocessing.Queue()

        blip_nlvr_visual_encoder_batches_queue=multiprocessing.Queue()
        blip_nlvr_visual_encoder_task_process=multiprocessing.Process(target=blip_nlvr_visual_encoder_task,args=(blip_nlvr_visual_encoder_batches_queue,blip_nlvr_visual_encoder_blip_nlvr_text_encoder_queue,))
        processes.append(blip_nlvr_visual_encoder_task_process)
        blip_nlvr_visual_encoder_task_process.start()

        blip_nlvr_text_encoder_batches_queue=multiprocessing.Queue()
        blip_nlvr_text_encoder_batches_return_queue=multiprocessing.Queue()
        blip_nlvr_text_encoder_task_process=multiprocessing.Process(target=blip_nlvr_text_encoder_task,args=(blip_nlvr_text_encoder_batches_queue,blip_nlvr_visual_encoder_blip_nlvr_text_encoder_queue,blip_nlvr_text_encoder_batches_return_queue,))
        processes.append(blip_nlvr_text_encoder_task_process)
        blip_nlvr_text_encoder_task_process.start()

        if fix_batch != False:
            blip_nlvr_visual_encoder_batch_size,blip_nlvr_text_encoder_batch_size=fix_batch
            print(f"fix batch sizes: {blip_nlvr_visual_encoder_batch_size},{blip_nlvr_text_encoder_batch_size}")
        
        while True:
            if fix_batch==False:
                try:
                    batch_sizes=batch_size_queue.get(block=False)
                    blip_nlvr_visual_encoder_batch_size,blip_nlvr_text_encoder_batch_size=batch_sizes
                    print(f"update batch sizes: {blip_nlvr_visual_encoder_batch_size},{blip_nlvr_text_encoder_batch_size}")
                except queue.Empty:
                    pass
            if not request_queue.empty():
                request_ids,batch_nums,images,texts,livings=request_queue.get()
                # print(f"request num: {len(request_ids)}")

                image0s=np.concatenate(image0s, axis=0)
                image1s=np.concatenate(image1s, axis=0)
                texts=np.concatenate(texts, axis=0)

                blip_nlvr_visual_encoder_batches_queue.put((image0s,image1s,blip_nlvr_visual_encoder_batch_size,livings),block=False)            

                blip_nlvr_text_encoder_batches_queue.put((texts,blip_nlvr_text_encoder_batch_size),block=False)

                if texts.shape[0]<=blip_nlvr_text_encoder_batch_size:
                    num_blip_nlvr_text_encoder_batch=1
                else:
                    num_blip_nlvr_text_encoder_batch=texts.shape[0]//blip_nlvr_text_encoder_batch_size
                    num_blip_nlvr_text_encoder_left=texts.shape[0]-num_blip_nlvr_text_encoder_batch*blip_nlvr_text_encoder_batch_size
                    if num_blip_nlvr_text_encoder_left!=0:
                        num_blip_nlvr_text_encoder_batch+=1
                
                batch_count=0
                now_left=None
                for _ in range(num_blip_nlvr_text_encoder_batch):
                    blip_nlvr_text_encoder_batches_return=blip_nlvr_text_encoder_batches_return_queue.get()
                    if now_left is None:
                        now_left=blip_nlvr_text_encoder_batches_return
                    else:
                        now_left=np.concatenate([now_left,blip_nlvr_text_encoder_batches_return],axis=0)
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

