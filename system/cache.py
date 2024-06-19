from collections import OrderedDict
import numpy as np
import queue
import time

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.start_time_cache = OrderedDict()
        self.living_cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            if time.time()-self.start_time_cache[key]<self.living_cache[key]:
                self.cache.move_to_end(key,last=True)
                self.start_time_cache[key]=time.time()
                self.start_time_cache.move_to_end(key,last=True)
                self.living_cache.move_to_end(key,last=True)
                return self.cache[key]  
            # in but timeout   
            del self.cache[key]
            del self.start_time_cache[key]
            del self.living_cache[key]
        return None

    def put(self, key, value,living=60*60*24*7):
        if living==-1:
            # no cache
            return
        if key not in self.cache and len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
            self.end_time_cache.popitem(last=False)

        self.cache[key] = value
        self.cache.move_to_end(key,last=True)
        self.start_time_cache[key]=time.time()
        self.start_time_cache.move_to_end(key,last=True)
        self.living_cache[key]=living
        self.living_cache.move_to_end(key,last=True)


def remove_duplication_cache(my_cache,images:np,livings:np):
    # print("remove_duplication_cache")
    # print("images.shape")
    # print(images.shape)
    # print("livings.shape")
    # print(livings.shape)
    # remove duplication in a batch
    unique_images,return_index = np.unique(images,return_index=True)
    # stable unique, keep the order in images
    sorted_indices = np.argsort(return_index)
    unique_images = unique_images[sorted_indices]
    livings=livings[sorted_indices]
    # print("livings.shape")
    # print(livings.shape)

    # np([aabb])->np([ab])

    # remove cached
    cached_images_result=[]
    no_forward_indices=[]
    for i,element in enumerate(unique_images):
        cached = my_cache.get(element)
        if cached is not None:
            cached_images_result.append(cached)
            no_forward_indices.append(i)
    
    if len(cached_images_result)!=0:
        cached_images_result=np.array(cached_images_result,dtype=cached_images_result[0].dtype)
    else:
        cached_images_result=np.array(cached_images_result)

    forward_indices=[i for i in range(unique_images.shape[0]) if i not in no_forward_indices]

    unique_images_forward = unique_images[forward_indices]

    livings_forward = livings[forward_indices]
    # print("livings.shape")
    # print(livings.shape)

    # np([ab])->np([a']),np([b])
    # unique->cached,forward
    # [012]->[0],[12]

    return unique_images,cached_images_result,np.array(no_forward_indices,dtype=int),unique_images_forward,np.array(forward_indices,dtype=int),livings_forward


def recover_once(images:np,unique_images:np,cached_images_result:np):
    recover_images_from_duplication=np.empty((images.shape[0],*(cached_images_result.shape[1:])),dtype=cached_images_result.dtype)
    for i,element in enumerate(cached_images_result):
        i_indices = np.where(images == unique_images[i])[0]
        recover_images_from_duplication[i_indices]=element
    return recover_images_from_duplication

def recover_cache_duplication(images:np,unique_images:np,cached_images_result:np,no_forward_indices:np,unique_images_after_forward:np,forward_indices:np,recover_images_from_duplication:np,mask:np):
    # print("forward_indices")
    # print(forward_indices)
    # print("no_forward_indices")
    # print(no_forward_indices)
    # print("unique_images_after_forward")
    # print(unique_images_after_forward.shape)
    # print("cached_images_result")
    # print(cached_images_result.shape)

    # recover from cache    
    if unique_images_after_forward.shape[0]!=0:
        recover_images_from_cache=np.empty((unique_images.shape[0],*(unique_images_after_forward.shape[1:])),dtype=unique_images_after_forward.dtype)
        recover_images_from_cache[forward_indices]=unique_images_after_forward
    else:
        recover_images_from_cache=np.empty((unique_images.shape[0],*(cached_images_result.shape[1:])),dtype=cached_images_result.dtype)
    
    if cached_images_result.shape[0]!=0:    
        recover_images_from_cache[no_forward_indices]=cached_images_result

    # recover from duplication
    for i in (forward_indices.tolist()+no_forward_indices.tolist()):
        i_indices = np.where(images == unique_images[i])[0]
        recover_images_from_duplication[i_indices]=recover_images_from_cache[i]
        mask[i_indices]=True

    return recover_images_from_duplication,mask

def cache_get_put(cache_get_input_queue,cache_get_output_queue,cache_put_queue,capacity=1000):
    try:
        image_cache=LRUCache(capacity)
        while True:
            try:
                images,livings=cache_get_input_queue.get(block=False)
                cache_get_output_queue.put(remove_duplication_cache(image_cache,images,livings),block=False)
            except queue.Empty:
                pass
            try:
                images_per_batch,output_per_batch,livings=cache_put_queue.get(block=False)
                for i,element in enumerate(output_per_batch):
                    image_cache.put(images_per_batch[i],element,livings[i])
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        pass

if __name__=="__main__":
    # c=OrderedDict()
    # c[2]=2
    # c[3]=3
    # c[5]=5
    # print(c)
    # # OrderedDict([(2, 2), (3, 3), (5, 5)])
    # c.move_to_end(2,last=True)
    # print(c)
    # # OrderedDict([(3, 3), (5, 5), (2, 2)])
    # c.popitem(last=True)
    # print(c)
    # # OrderedDict([(3, 3), (5, 5)])
    
    # type(b"aa")
    # <class 'bytes'>
    # print(b"sss" is b"sss")
    # True
    # print(id(b"ssss"),id(b"ssss"))
    # True
    # Two bytes objects are considered equal if and only if they have the same length and each byte at corresponding positions is the same.

    capacity=100
    my_cache=LRUCache(capacity)

    for _ in range(2):
        batch_size=5*2
        images = [
            "/workspace/demos/images/merlion.png","/workspace/demos/images/beach.jpg"
        ]*(batch_size//2)
        texts = [
            "where is it?","where is the woman sitting?"
        ]*(batch_size//2)
        images=np.array([image.encode('utf-8') for image in images])
        texts=np.array([text.encode('utf-8') for text in texts])

        unique_images,cached_images_result,no_forward_indices,unique_images_forward,forward_indices=remove_duplication_cache(my_cache,images)
        print("images")
        print(images)
        print("unique_images")
        print(unique_images)
        print("cached_images_result")
        print(cached_images_result)
        print("unique_images_forward")
        print(unique_images_forward)

        # forward
        unique_images_after_forward=np.array([[i] for i in range(unique_images_forward.shape[0])])
        print("unique_images_after_forward")
        print(unique_images_after_forward)

        # add cache
        for i,element in enumerate(unique_images_after_forward):
            my_cache.put(unique_images_forward[i],element)

        if unique_images_after_forward.shape[0]!=0:
            recover_images_from_duplication=np.empty((images.shape[0],*(unique_images_after_forward.shape[1:])),dtype=unique_images_after_forward.dtype)
        else:
            recover_images_from_duplication=np.empty((images.shape[0],*(cached_images_result.shape[1:])),dtype=cached_images_result.dtype)
        mask=np.full(images.shape[0],False)
        recover_images_from_duplication,mask=recover_cache_duplication(images,unique_images,cached_images_result,no_forward_indices,unique_images_after_forward,forward_indices,recover_images_from_duplication,mask)

        print("recover_images_from_duplication")
        print(recover_images_from_duplication)
        print("mask")
        print(mask)
        print("-------------------------------------------------------------------")