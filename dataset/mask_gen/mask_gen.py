import sys
sys.path.append("../../")
from .common import *
from utils import *

def to_numpy(tensor_list):
    return [x.cpu().numpy() for x in tensor_list]

def gen_mask_core(image,center=np.array([72,62,64]),jitter=8,rmin_h=35,rmax_h=60,rmin_w=25,rmax_w=50,rmin_l=35,rmax_l=60):
    #image is of 4-dimension shape
    rmax_h=min(max(rmax_h,rmin_h+1),126)
    rmax_w=min(max(rmax_w,rmin_w+1),126)
    rmax_l=min(max(rmax_l,rmin_l+1),126)
    jit_center=center+np.random.randint(low=-jitter,high=jitter+1,size=3)
    range_h=np.random.randint(low=rmin_h,high=rmax_h)
    range_w=np.random.randint(low=rmin_w,high=rmax_w)
    range_l=np.random.randint(low=rmin_l,high=rmax_l)
    end_max=128-60
    start_h=np.clip(jit_center[0]-range_h//2,0,end_max)    
    end_h=start_h+range_h
    start_w=np.clip(jit_center[1]-range_w//2,0,end_max) - 3
    end_w=start_w+range_w
    start_l=np.clip(jit_center[2]-range_l//2,0,end_max)
    end_l=start_l+range_l
    mask=np.zeros_like(image)
    mask[:,start_h:end_h+1,start_w:end_w+1,start_l:end_l+1]=1
    return mask

def gen_mask(image,gen_unilateral=True):
    if(gen_unilateral):
        mask=gen_mask_core(image,rmin_l=35,rmax_l=60)
    else:
        mask=gen_mask_core(image,rmin_l=60,rmax_l=90)
    return mask

def gen_determine_mask(image,h,w,l,center=np.array([72,56,64])):
    eps=1e-5
    mask=gen_mask_core(image,center=center,jitter=0,rmin_h=h,rmax_h=h+eps,rmin_w=w,rmax_w=w+eps,rmin_l=l,rmax_l=l+eps)
    return mask

def gen_mask_batch(image_batch,gen_unilateral=True):
    batch_num=image_batch.shape[0]
    mask_list=[]
    for batch_idx in range(0,batch_num):
        mask=gen_mask(image_batch[batch_idx],gen_unilateral)
        mask_list.append(mask)
    return torch.stack(mask_list)
