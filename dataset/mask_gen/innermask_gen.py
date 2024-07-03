import sys
sys.path.append("../../")
from .common import *
from utils import *

def gen_innermask(image,mask,thresh=1500/2300):
    innermask=torch.where(image>thresh,1.0,0.0)*mask
    return innermask

def gen_innermask_batch(image_batch,mask_batch,thresh=(1500+1000)/2300):
    batch_num=image_batch.shape[0]
    innermask_list=[]
    for batch_idx in range(0,batch_num):
        innermask=gen_innermask(image_batch[batch_idx],mask_batch[batch_idx],thresh=thresh)
        innermask_list.append(innermask)
    return torch.stack(innermask_list)

def gen_determine_innermask(image,mask,thresh=1500/2300):
    #input image, mask [c,h,w,l] tensor
    #output innermask [c,h,w,l] tensor
    innermask=gen_innermask(image,mask,thresh=thresh)
    return innermask

def gen_determine_innermask_batch(image_batch,mask_batch,thresh=1500/2300):
    batch_num = image_batch.shape[0]
    innermask_list=[]
    for bacth_idx in range(0,batch_num):
        innermask=gen_determine_innermask(image_batch[bacth_idx],mask_batch[bacth_idx],thresh=thresh)
        innermask_list.append(innermask)
    return torch.stack(innermask_list)