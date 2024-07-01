import sys
sys.path.append("../")
from dependencies import *
from utils import *
from ..dependencies import *
from .volumentations import ElasticTransform,Compose
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation

def norm_vec(vec):
    norm=np.linalg.norm(vec)
    return vec/norm

def cal_point_dist(point_a, point_b):
    point_dist = np.sqrt(np.sum((point_a-point_b)**2))
    return point_dist

def move_left(cube):
    _,_,h,w,l=cube.shape
    move_cube=torch.zeros_like(cube)
    move_cube[:,:,:,:,0:l-1]=cube[:,:,:,:,1:l]
    return move_cube

def move_right(cube):
    _,_,h,w,l=cube.shape
    move_cube=torch.zeros_like(cube)
    move_cube[:,:,:,:,1:l]=cube[:,:,:,:,0:l-1]
    return move_cube

def move_front(cube):
    _,_,h,w,l=cube.shape
    move_cube=torch.zeros_like(cube)
    move_cube[:,:,:,0:w-1,:]=cube[:,:,:,1:w,:]
    return move_cube

def move_back(cube):
    _,_,h,w,l=cube.shape
    move_cube=torch.zeros_like(cube)
    move_cube[:,:,:,1:w,:]=cube[:,:,:,0:w-1,:]
    return move_cube

def move_down(cube):
    _,_,h,w,l=cube.shape
    move_cube=torch.zeros_like(cube)
    move_cube[:,:,0:h-1,:,:]=cube[:,:,1:h,:,:]
    return move_cube

def move_up(cube):
    _,_,h,w,l=cube.shape
    move_cube=torch.zeros_like(cube)
    move_cube[:,:,1:h,:,:]=cube[:,:,0:h-1,:,:]
    return move_cube

def get_init_kernel3():
    init_weight=torch.zeros(size=[1,1,3,3,3])
    init_weight[0,0,1,1,1]=1
    random_weight=torch.rand(size=[1,1,3,3,3])
    final_weight=torch.where((init_weight+random_weight)>0.5,1.0,0.0)
    return final_weight

def get_init_kernel5():
    init_weight=torch.zeros(size=[1,1,5,5,5])
    init_weight[0,0,1:4,1:4,1:4]=1
    random_weight=torch.rand(size=[1,1,5,5,5])
    final_weight=torch.where((init_weight+random_weight)>0.5,1.0,0.0)
    return final_weight

def get_full_kernel3():
    return torch.ones(size=[1,1,3,3,3])

def get_full_kernel5():
    return torch.ones(size=[1,1,5,5,5])

def get_full_kernel7():
    return torch.ones(size=[1,1,7,7,7])

def random_transform_3D(image,
                        iter_num=3,
                        sigma=(1.0, 1.0, 1.0),
                        max_deform=(3.0, 3.0, 3.0),
                        deform_limits=(0.3,1),
                        interpolation=0,
                        margin=8):
    trans_image=copy.copy(image)
    image_h, image_w, image_l = image.shape
    if(np.sum(image)>0):
        hs, ws, ls = np.where(image>0)
        min_h, max_h = np.min(hs)-margin, np.max(hs)+margin
        min_w, max_w = np.min(ws)-margin, np.max(ws)+margin
        min_l, max_l = np.min(ls)-margin, np.max(ls)+margin
        max_h = np.clip(max_h, 0, image_h-1)
        min_h = np.clip(min_h, 0, max_h)
        max_w = np.clip(max_w, 0, image_w-1)
        min_w = np.clip(min_w, 0, max_w)
        max_l = np.clip(max_l, 0, image_l-1)
        min_l = np.clip(min_l, 0, max_l)
        trans_cube = image[min_h:max_h+1,min_w:max_w+1,min_l:max_l+1]
        transform = Compose(
                [ElasticTransform(always_apply=True,deformation_limits=deform_limits,p=1,interpolation=interpolation)]
            )
        for i in range(0,iter_num):
            trans_cube=transform(**{"image":trans_cube})["image"]
        trans_image[min_h:max_h+1,min_w:max_w+1,min_l:max_l+1]=trans_cube
    trans_image=np.round(trans_image).astype(np.int32)
    return trans_image

def random_transform_4D(image,
                        iter_num=3,
                        sigma=(1.0, 1.0, 1.0),
                        max_deform=(3.0, 3.0, 3.0),
                        deform_limits=(0.3,1),
                        interpolation=0):
    return random_transform_3D(image[0],
                               iter_num=iter_num,
                               sigma=sigma,
                               max_deform=max_deform,
                               deform_limits=deform_limits,
                               interpolation=interpolation)[np.newaxis,:,:,:]

def gen_defactimage(image,maximask,cleftmask,thresh_air=800/2300,fill_value=(1100+1000)/2300,gen_by_slice=True,eps=0.1):
    #input 4-dimension image
    with torch.no_grad():
        #gen filter_mask
        spread_weight=get_full_kernel7()
        spread_cleftmask=F.conv3d(input=torch.FloatTensor(cleftmask[np.newaxis,:,:,:,:]),weight=spread_weight,stride=1,padding=3)
        spread_cleftmask=torch.where(spread_cleftmask[0]>eps,1,0)
        neighbor_cleftmask=torch.where(torch.logical_and(spread_cleftmask!=0,maximask==0),1,0)
        invairhole_mask=torch.where(image>thresh_air,1,0)
        filter_mask=torch.logical_and(neighbor_cleftmask>0.1,invairhole_mask>0.1)
        fill_mask=torch.where(torch.logical_or(cleftmask!=0,filter_mask!=0),1,0)
        #gen images
        defactimage=image.clone()
        if(gen_by_slice):
            delta_slice=3
            _,hs,_,_=torch.where(cleftmask!=0)
            min_h,max_h=torch.min(hs),torch.max(hs)
            for slice_h in range(min_h,max_h+1):
                filter_mask_slice=filter_mask[:,slice_h:slice_h+delta_slice,:,:]
                image_slice=image[:,slice_h:slice_h+delta_slice,:,:]
                fill_mask_slice=fill_mask[:,slice_h:slice_h+delta_slice,:,:]
                fill_num=torch.sum(filter_mask_slice)
                if(fill_num<10):
                    defactimage[:,slice_h:slice_h+delta_slice]=image_slice*(1-fill_mask_slice)+fill_value*fill_mask_slice
                else:
                    filter_values,_=image_slice[filter_mask_slice].flatten().sort()
                    filter_values=filter_values[int(filter_values.shape[0]*0.3):int(filter_values.shape[0]*0.7)]
                    fill_idxs=(torch.rand_like(fill_mask_slice.float())*(filter_values.shape[0]-1)).long()
                    fill_slice=filter_values[fill_idxs.flatten()].reshape(fill_mask_slice.shape)
                    fill_slice=fill_slice*fill_mask_slice+image_slice*(1-fill_mask_slice)
                    fill_slice=torch.FloatTensor(gaussian_filter(fill_slice.cpu(),sigma=0.4)).to(image_slice.device)
                    defactimage[:,slice_h:slice_h+delta_slice]=image_slice*(1-fill_mask_slice)+fill_slice*fill_mask_slice
        return defactimage
    
def gen_defactimage_fillrec(image,maximask,cleftmask,thresh_air,fill_value=(1100+1000)/2300,gen_by_slice=True,fill_air=False,eps=0.1):
    #input 4-dimension image
    with torch.no_grad():
        #gen filter_mask
        spread_weight=get_full_kernel7()
        spread_cleftmask=F.conv3d(input=torch.FloatTensor(cleftmask[np.newaxis,:,:,:,:]),weight=spread_weight,stride=1,padding=3)
        spread_cleftmask=torch.where(spread_cleftmask[0]>eps,1,0)
        neighbor_cleftmask=torch.where(torch.logical_and(spread_cleftmask!=0,maximask==0),1,0)
        invairhole_mask=torch.where(image>thresh_air,1,0)
        filter_mask=torch.logical_and(neighbor_cleftmask>0.1,invairhole_mask>0.1)
        fill_mask=torch.where(torch.logical_or(cleftmask!=0,filter_mask!=0),1,0)
        fill_mask_origin=fill_mask.clone()
        if(fill_air):
            fill_mask=get_mask_bbx4D_tensor(fill_mask,2,2,2)*(1-maximask*(1-fill_mask))
        else:
            fill_mask=get_mask_bbx4D_tensor(fill_mask,2,2,2)*(1-maximask*(1-fill_mask))*invairhole_mask
        #gen images
        defactimage=image.clone()
        if(gen_by_slice):
            delta_slice=3
            _,hs,_,_=torch.where(cleftmask!=0)
            min_h,max_h=torch.min(hs),torch.max(hs)
            for slice_h in range(min_h,max_h+1):
                filter_mask_slice=filter_mask[:,slice_h:slice_h+delta_slice,:,:]
                image_slice=image[:,slice_h:slice_h+delta_slice,:,:]
                fill_mask_slice=fill_mask[:,slice_h:slice_h+delta_slice,:,:]
                fill_num=torch.sum(filter_mask_slice)
                if(fill_num<10):
                    defactimage[:,slice_h:slice_h+delta_slice]=image_slice*(1-fill_mask_slice)+fill_value*fill_mask_slice
                else:
                    filter_values,_=image_slice[filter_mask_slice].flatten().sort()
                    filter_values=filter_values[int(filter_values.shape[0]*0.3):int(filter_values.shape[0]*0.7)]
                    fill_idxs=(torch.rand_like(fill_mask_slice.float())*(filter_values.shape[0]-1)).long()
                    fill_slice=filter_values[fill_idxs.flatten()].reshape(fill_mask_slice.shape)
                    fill_slice=fill_slice*fill_mask_slice+image_slice*(1-fill_mask_slice)
                    fill_slice=torch.FloatTensor(gaussian_filter(fill_slice.cpu(),sigma=0.4)).to(image_slice.device)
                    defactimage[:,slice_h:slice_h+delta_slice]=image_slice*(1-fill_mask_slice)+fill_slice*fill_mask_slice
        return defactimage

class PoissonSoftTissueFiller:
    def __init__(self,
                 invalid_mask_black_thresh=0.3,
                 invalid_mask_light_thresh=0.6,
                 dilate_border=5,
                 use_modi_border_cond=True,
                 use_fill_contour=True,
                 use_fill_soft_periphery=True,
                 log_path=None,
                 ret_middle_result=False,
                 ref_soft_tissue_all_image_path=None,
                 ref_soft_tissue_all_seg_path=None,
                 debug=False
                 ):
        self.invalid_mask_black_thresh = invalid_mask_black_thresh
        self.invalid_mask_light_thresh = invalid_mask_light_thresh
        self.dilate_border = dilate_border
        self.use_modi_border_cond = use_modi_border_cond
        self.use_fill_contour = use_fill_contour
        self.use_fill_soft_periphery = use_fill_soft_periphery
        self.log_path = log_path
        self.ret_middle_result = ret_middle_result
        self.ref_soft_tissue_all_image_path = ref_soft_tissue_all_image_path
        self.ref_soft_tissue_all_seg_path = ref_soft_tissue_all_seg_path
        self.debug = debug
        
        if(self.log_path is not None):
            make_parent_dir(self.log_path)
            init_log(self.log_path)
            
        self.log_func("[PoissonSoftTissueFiller]")
        self.log_func(f"\t use_modi_border_cond: {self.use_modi_border_cond}")
        self.log_func(f"\t use_fill_contour: {self.use_fill_contour}")
        self.log_func(f"\t use_fill_soft_periphery: {self.use_fill_soft_periphery}")
        self.log_func(f"\t ref_soft_tissue_all_image_path: {self.ref_soft_tissue_all_image_path}")
        self.log_func(f"\t ref_soft_tissue_all_seg_path: {self.ref_soft_tissue_all_seg_path}")
        

    def debug_print(self, msg):
        if(self.debug):
            print(msg)

    def log_func(self, msg):
        if(self.log_path is not None):
            log(msg)
        else:
            print(msg)

    def get_invalid_mask(self,
                         ref_image,
                         ref_maxi_seg=None
                        ):
        invalid_mask = np.zeros_like(ref_image)
        # maxi mask
        if(ref_maxi_seg is not None):
            invalid_mask = invalid_mask + ref_maxi_seg
        # black mask
        black_mask = np.where(ref_image<self.invalid_mask_black_thresh, 1, 0).astype(np.float32)
        invalid_mask += black_mask
        # white mask
        light_mask = np.where(ref_image>self.invalid_mask_light_thresh, 1, 0).astype(np.float32)
        invalid_mask += light_mask
        # fuse
        invalid_mask = np.where(invalid_mask>1e-1, 1, 0).astype(np.float32)
        return invalid_mask

    def search_valid_loc(self,
                         syn_cleft,
                         ref_image,
                         ref_maxi_seg=None
                         ):
        """
        search valid location of the reference image to crop the soft tissue and fill the synthetic cleft.

        input:
            syn_cleft:  
                array, [h,w,l] binary. binary mask of synthetic cleft.
            ref_image:  
                array, [h,w,l] [0,1] float. reference image to find soft tissue to fill in.
            ref_maxi_seg:  
                array, [h,w,l] [0,1] binary. the maxi segment of the reference image.

        output:
            valid_loc_dict, contains
                    valid_loc_list:  
                        List, a list of valid loc,
                    invalid_mask:  
                        array, [h,w,l] binary, invalid region mask used for searching valid loc,
                    cleft_crop_info:  
                        CropInfo3D object, contains the crop index of syn_clefy to generate the kernel.
        """
        self.debug_print("[Search Valid Loc]")
        dilate_syn_cleft_arr = binary_dilation(input=syn_cleft, iterations=self.dilate_border).astype(np.float32)
        crop_bbxmask = get_mask_bbx3D_array(label_cube=dilate_syn_cleft_arr, margin=0)
        kernel_arr, cleft_crop_info = get_crop_image_by_bbx_mask_array3D(image=dilate_syn_cleft_arr, 
                                                                         bbx_mask=crop_bbxmask,
                                                                         margin=4)
        invalid_mask_arr = self.get_invalid_mask(ref_image=ref_image, ref_maxi_seg=ref_maxi_seg)
        kernel = torch.FloatTensor(kernel_arr).unsqueeze(0).unsqueeze(0)
        invalid_mask = torch.FloatTensor(invalid_mask_arr).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            conv_result = F.conv3d(input=invalid_mask, weight=kernel, stride=1, padding=0)
            if(self.debug):
                sum_conv_result = torch.sum(torch.where(conv_result<1e-1, 1, 0))
                sum_kernel = torch.sum(kernel)
                self.debug_print(f"sum_conv_result: {sum_conv_result} sum_kernel:{sum_kernel}")
        valid_loc_list = np.where(conv_result.squeeze().numpy()==0)
        valid_loc_list = list(zip(valid_loc_list[0], valid_loc_list[1], valid_loc_list[2]))
        valid_flag = (len(valid_loc_list)!=0)
        valid_loc_dict = {
            "valid_flag" : valid_flag,
            "valid_loc_list" : valid_loc_list,
            "cleft_crop_info" : cleft_crop_info
        }
        if(self.ret_middle_result):
            valid_loc_dict["invalid_mask"] = invalid_mask_arr
        return valid_loc_dict

    def search_valid_loc_on_paths(self,
                        syn_cleft,
                        name_paths_dict,
                        syn_name=""
                        ):
        """
        search on all the paths, to find a valid location of the reference image to crop the soft tissue and fill the synthetic cleft.
        
        can use the bone voxel of the front maxi to fill the sythetic cleft bone border, determined by use_fill_contour flag
        
        input:    
            syn_cleft 
                array, [h,w,l] binary, synthetic cleft binary mask.
            name_paths_dict 
                dict, key: 'name' value: a dict with key:'image_path', 'maxi_path'
        
        output:
            fuse_image
                array, [h,w,l] float, the synthetic image that use the soft-tissue of the reference images(noted from paths) to fill the synthetic cleft region.

        notes:
            dst_image need to be in [0,1] float
        """
        self.debug_print("[Search on Paths]")
        ref_name_list = copy.copy(list(name_paths_dict.keys()))
        total_ref_num = len(ref_name_list)
        np.random.shuffle(ref_name_list)
        for try_idx, name in enumerate(ref_name_list):
            self.debug_print(f"\t search on paths tring {try_idx+1}/{total_ref_num}")
            path_dict = name_paths_dict[name]
            # ref image
            ref_image_path = path_dict["image_path"]
            ref_image = read_mha_array3D(ref_image_path)
            # ref maxi
            if("maxi_path" in path_dict):
                ref_maxi_path = path_dict["maxi_path"]
                ref_maxi_seg = read_mha_array3D(ref_maxi_path)
            else:
                ref_maxi_seg = None
                
            # search valid loc
            valid_loc_dict = self.search_valid_loc(
                syn_cleft = syn_cleft,
                ref_image = ref_image,
                ref_maxi_seg = ref_maxi_seg
            )
            
            valid_loc_list = valid_loc_dict["valid_loc_list"]
        
            # no valid loc
            if(len(valid_loc_list)==0):
                continue
            # found valid loc, perform poisson-edit
            valid_loc = not_putback_sampling(valid_loc_list, sample_num=1)[0]
            self.log_func(f"time:{get_cur_time_str()} pid:{os.getpid()} Syn {syn_name}, Found soft tissue {try_idx+1}/{total_ref_num} on {name}, loc: {valid_loc}, border: {self.dilate_border}")
            # add value
            search_result_dict = valid_loc_dict
            search_result_dict["valid_loc"] = valid_loc
            search_result_dict["image_path"] = ref_image_path
            search_result_dict["maxi_path"] = ref_maxi_path
            search_result_dict["search_process"] = f"{try_idx}/{total_ref_num}"
            return search_result_dict
        if((self.ref_soft_tissue_all_image_path is not None) and (self.ref_soft_tissue_all_seg_path is not None)):
            self.log_func(f"time:{get_cur_time_str()} pid:{os.getpid()} Syn {syn_name} not found Soft-Tissue Reference Image, use Soft-Tissue-All Reference image")
            ref_soft_tissue_all_image = read_mha_array3D(self.ref_soft_tissue_all_image_path)
            ref_soft_tissue_all_seg = read_mha_array3D(self.ref_soft_tissue_all_seg_path)
            valid_loc_dict = self.search_valid_loc(
                syn_cleft = syn_cleft,
                ref_image = ref_soft_tissue_all_image,
                ref_maxi_seg = ref_soft_tissue_all_seg
            )
            # add value
            valid_loc_list = valid_loc_dict["valid_loc_list"]
            valid_loc = not_putback_sampling(valid_loc_list, sample_num=1)[0]
            search_result_dict = valid_loc_dict
            search_result_dict["valid_loc"] = valid_loc
            search_result_dict["image_path"] = self.ref_soft_tissue_all_image_path
            search_result_dict["maxi_path"] = self.ref_soft_tissue_all_seg_path
            search_result_dict["search_process"] = f"use soft tissue"
            return search_result_dict
        else:
            self.log_func(f"time:{get_cur_time_str()} pid:{os.getpid()} Syn {syn_name}, not found Soft-Tissue Reference Image and Doesn't have Soft-Tissue-All Reference image, Error!")
            raise NotImplementedError(f"error not found valid soft-tissue template image: {syn_name}")
    
    def fill_bone_border(self,
                tgt_image,
                tgt_seg_maxi,
                tgt_syn_cleft,
                fill_image,
                bone_border,
                low_bone_thresh = 0.85,
                high_bone_thresh = 1.05
                ):
        """
        fill bone border using maxi bone-tissue voxel value sampling, gaussian filter and possion edition.
        """
        # gen bone texture
        maxi_seg_front = get_mask_front_contour_array3D(mask=tgt_seg_maxi)
        maxi_seg_front = np.where(tgt_image<low_bone_thresh, 0, maxi_seg_front)
        maxi_seg_front = np.where(tgt_image>high_bone_thresh, 0, maxi_seg_front)
        maxi_seg_front_sel = np.where(maxi_seg_front>1e-1, True, False)
        bone_candidate=  tgt_image[maxi_seg_front_sel]
        bone_texture = np.random.choice(bone_candidate, size=fill_image.shape)
        bone_texture = gaussian_filter(bone_texture, sigma=1)
        # gen fill mask
        fill_mask = copy.copy(bone_border)
        fill_mask = binary_dilation(fill_mask, iterations=1)
        fill_mask = np.where(tgt_seg_maxi>1e-1, fill_mask, 0).astype(np.float32)
        fill_mask = np.where(tgt_syn_cleft<1e-1, fill_mask, 0).astype(np.float32)
        fill_mask_sel = np.where(fill_mask>1e-1, True, False)
        # gen naive fill image
        naive_fill_image = copy.copy(fill_image)
        naive_fill_image[fill_mask_sel] = bone_texture[fill_mask_sel]
        # gen gaussian fill image
        gaussian_fill_image = gaussian_filter(input=naive_fill_image, sigma=0.6)
        # gen poission fill image
        possion_fill_image = possion_image_edit_array3d(
            src_image = gaussian_fill_image,
            dst_image = naive_fill_image,
            mask = fill_mask
        )
        return possion_fill_image

    def fill_soft_tissue(self,
                        tgt_image,
                        tgt_seg_maxi,
                        tgt_syn_cleft,
                        tgt_syn_cleft_crop_info,
                        ref_image,
                        ref_seg_maxi,
                        ref_valid_loc,
                        ret_middle_result=False
                        ):
        """
        input:
            com_image
                array, [h,w,l] [0,1] float, a normal image with complete maxi.
            syn_cleft
                array, [h,w,l] binary, the synthetic cleft label of the normal image.
            maxi_seg
                array, [h,w,l] binary, the maxi seg of the normal image.
            ref_paths_dict 
                dict, key: 'name' value: a dict with key:'image_path', 'maxi_path'.
        
        output:

        
        notes:
            dst_image need to be in [0,1] float
        """        
        # gen periphery
        cleft_periphery = get_mask_periphery_array3D(tgt_syn_cleft)
        bone_periphery = np.where(np.logical_and(tgt_seg_maxi>1e-1, cleft_periphery>1e-1), 1, 0)
        soft_periphery = np.where(np.logical_and(bone_periphery<1e-1, cleft_periphery>1e-1), 1, 0)
        
            
        # transfer the src_image to proper location to get its soft-tissue corresponding to the synthetic cleft
        cleft_sx, cleft_sy, cleft_sz = tgt_syn_cleft_crop_info.min_h, tgt_syn_cleft_crop_info.min_w, tgt_syn_cleft_crop_info.min_l
        tissue_sx, tissue_sy, tissue_sz = ref_valid_loc
        src_image = spatial_transform_by_param_array3D(
            image = ref_image,
            trans_x = -tissue_sx + cleft_sx,
            trans_y = -tissue_sy + cleft_sy,
            trans_z = -tissue_sz + cleft_sz,
            mode = "bilinear"
        )
        
        # use possion edition to fill the soft-tissue on the synthetic cleft part
        self.debug_print("[Poisson Edit]")
        # modify the possion edit condition value to generate more lifelike soft-tissue
        if(self.use_modi_border_cond):
            fill_image = possion_image_edit_by_condmask_array3d(
                src_image = src_image,
                dst_image = tgt_image,
                mask = tgt_syn_cleft,
                src_cond_mask = bone_periphery,
                tgt_cond_mask = soft_periphery
            )
        else:
            # just use the original possion edit conditon
            fill_image = possion_image_edit_array3d(
                src_image = src_image,
                dst_image = tgt_image,
                mask = tgt_syn_cleft
            )
        
        # fill bone-border
        if(self.use_fill_contour):
            fill_image = self.fill_bone_border(
                tgt_image = tgt_image,
                tgt_seg_maxi = tgt_seg_maxi,
                tgt_syn_cleft = tgt_syn_cleft,
                fill_image = fill_image,
                bone_border = bone_periphery
            )

        return fill_image
    
    def fill_soft_tissue_by_searching_paths(self,
                                            tgt_image,
                                            tgt_seg_maxi,
                                            tgt_syn_cleft,
                                            name_paths_dict,
                                            syn_name="",
                                            ret_middle_result=False):
        """
        fill soft tissue by searching on all the image paths in 'name_paths_dict', 
        
        input array3D
        """
        
        # gen periphery
        cleft_periphery = get_mask_periphery_array3D(tgt_syn_cleft, border_width=2)
        bone_periphery = np.where(np.logical_and(tgt_seg_maxi>1e-1, cleft_periphery>1e-1), 1, 0)
        soft_periphery = np.where(np.logical_and(bone_periphery<1e-1, cleft_periphery>1e-1), 1, 0)
        
        fill_syn_cleft = np.where(soft_periphery>1e-1, soft_periphery, tgt_syn_cleft)
        
        arg_syn_cleft = tgt_syn_cleft
        if(self.use_fill_soft_periphery):
            arg_syn_cleft = fill_syn_cleft
        
        # search valid location of soft-tissue
        search_result_dict = self.search_valid_loc_on_paths(
            syn_cleft = arg_syn_cleft,
            name_paths_dict = name_paths_dict,
            syn_name = syn_name
        )
        
        tgt_syn_cleft_crop_info = search_result_dict["cleft_crop_info"]
        ref_image_path = search_result_dict["image_path"]
        ref_seg_maxi_path = search_result_dict["maxi_path"]
        ref_valid_loc = search_result_dict["valid_loc"]
        
        ref_image = read_mha_array3D(ref_image_path)
        ref_seg_maxi = read_mha_array3D(ref_seg_maxi_path)
        
        fill_image = self.fill_soft_tissue(
            tgt_image = tgt_image,
            tgt_seg_maxi = tgt_seg_maxi,
            tgt_syn_cleft = arg_syn_cleft,
            tgt_syn_cleft_crop_info = tgt_syn_cleft_crop_info,
            ref_image = ref_image,
            ref_seg_maxi = ref_seg_maxi,
            ref_valid_loc = ref_valid_loc
        )
        return fill_image
