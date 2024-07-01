import torch
def get_local_patch(x,mask,patch_h=64,patch_w=64,patch_l=64):
    #get local patch of the binary mask
    #input [c,h,w,l] tensor
    _,h,w,l=mask.shape
    if(torch.sum(mask)!=0):
        _,hs,ws,ls=torch.where(mask!=0)
        min_h,max_h=torch.min(hs),torch.max(hs)
        min_w,max_w=torch.min(ws),torch.max(ws)
        min_l,max_l=torch.min(ls),torch.max(ls)
        start_h=min_h
        start_w=min_w
        start_l=min_l
        if(start_h+patch_h>=h):
            start_h=max_h-patch_h
        if(start_w+patch_w>=w):
            start_w=max_w-patch_w
        if(start_l+patch_l>=l):
            start_l=max_l-patch_l
    else:
        start_h=h-patch_h//2
        start_w=w-patch_w//2
        start_l=l-patch_l//2
    patch_x=x[:,start_h:start_h+patch_h,start_w:start_w+patch_w,start_l:start_l+patch_l]
    patch_mask=mask[:,start_h:start_h+patch_h,start_w:start_w+patch_w,start_l:start_l+patch_l]
    return patch_x,patch_mask

def get_local_patch_batch(x,mask,patch_h=64,patch_w=64,patch_l=64):
    #get local patch of the binary mask
    #input [b,c,h,w,l] tensor
    b_num=x.shape[0]
    patch_x_list=[]
    patch_mask_list=[]
    for b_idx in range(0,b_num):
        patch_x,patch_mask=get_local_patch(x[b_idx],mask[b_idx],patch_h=patch_h,patch_w=patch_w,patch_l=patch_l)
        patch_x_list.append(patch_x)
        patch_mask_list.append(patch_mask)
    patch_x_batch=torch.stack(patch_x_list)
    patch_mask_batch=torch.stack(patch_mask_list)
    return patch_x_batch,patch_mask_batch