import sys
sys.path.append("../../")
from utils import *
from .common import *
from .mask_gen import *
from .innermask_gen import *

class ImitatemaskGenerator:
    def __init__(self,
                 thresh_bone=1350/2300,
                 thresh_air=800/2300,
                 gen_unilateral_possibility = 0.7,
                 use_random_transform_skeleton = True,
                 use_random_transform_dilate = True,
                 use_gradual_grow = True,
                 use_clean_chips = True,
                 use_clean_chips_all = False,
                 start_from_center_bias_h = -15,
                 start_from_center_bias_w = -31,
                 start_from_center_bias_l = -18,
                 start_from_close_bias_w = 8,
                 start_random_h = 2,
                 start_random_w = 2,
                 start_random_l = 2,
                 dst_from_start_bias_h = 15,
                 dst_from_start_bias_w = 5,
                 dst_from_start_bias_l = 3,
                 dst_random_h = 2,
                 dst_random_w = 2,
                 dst_random_l = 1,
                 dilate_random_low = 3,
                 dilate_random_high = 5,
                 step_random_low = 40,
                 step_random_high = 70,
                 skeleton_delta_size_per_step = 0.15,
                 skeleton_reachdst_size_per_step = 0.05,
                 random_transform_limit_low = 0.15,
                 random_transform_limit_high = 0.45,
                 init_cube_size=20,
                 debug=False):
        self.debug = debug
        # thresh
        self.thresh_bone = thresh_bone
        self.thresh_air = thresh_air

        # start from center bias, use add
        # bias-l is for left
        self.start_from_center_bias_h = start_from_center_bias_h
        self.start_from_center_bias_w = start_from_center_bias_w
        self.start_from_center_bias_l = start_from_center_bias_l
        self.start_from_close_bias_w = start_from_close_bias_w
        # start random bias, use add
        self.start_random_h = start_random_h
        self.start_random_w = start_random_w
        self.start_random_l = start_random_l

        # dst from start bias, use add
        # bias-l is for left
        self.dst_from_start_bias_h = dst_from_start_bias_h
        self.dst_from_start_bias_w = dst_from_start_bias_w
        self.dst_from_start_bias_l = dst_from_start_bias_l
        # dst random bias, use add
        self.dst_random_h = dst_random_h
        self.dst_random_w = dst_random_w
        self.dst_random_l = dst_random_l

        # skeleton gen
        # unilateral
        self.gen_unilateral_possibility = gen_unilateral_possibility
        # step num
        self.step_random_low = step_random_low
        self.step_random_high = step_random_high
        # delta step
        self.init_cube_size = init_cube_size # 10
        self.skeleton_delta_size_per_step = skeleton_delta_size_per_step
        self.skeleton_reachdst_size_per_step = skeleton_reachdst_size_per_step
        # random transform augment
        self.use_random_transform_skeleton = use_random_transform_skeleton
        self.use_random_transform_dilate = use_random_transform_dilate
        self.random_transform_limit_low = random_transform_limit_low
        self.random_transform_limit_high = random_transform_limit_high
        # gradual grow augment
        self.use_gradual_grow = use_gradual_grow
        # clean ships
        self.use_clean_chips = use_clean_chips
        self.use_clean_chips_all = use_clean_chips_all
        
        # cleftmask dilate
        self.dilate_random_low = dilate_random_low
        self.dilate_random_high = dilate_random_high
        
        print("[Imitatemask Generator]")
        print("Check Status")
        print(f"\tDebug: {self.debug}")
        print(f"\tUse Random Transform Skeleton: {self.use_random_transform_skeleton}")
        print(f"\tUse Random Transform Dilate: {self.use_random_transform_dilate}")
        print(f"\tUse Gradual Grow: {self.use_gradual_grow}")
        print(f"\tUse Clean Chips: {self.use_clean_chips}")
        print(f"\tUse Clean Chips All: {self.use_clean_chips_all}")
    
    def debug_print(self,msg,flag=False):
        if(self.debug or flag):
            print(msg)
    
    # below are core generator functions
    def gen_pos_random(self, innermask, side_sign):
        """
            innermask: array4D, [c,h,w,l]
            side_sign:
                left side: 1
                right side: -1
        """
        _,hs,ws,ls = np.where(innermask!=0)
        center_h, center_w, center_l = np.mean(hs), np.mean(ws), np.mean(ls)
        # gen skeleton start
        # start h
        start_random_bias_h = np.random.uniform(low=-self.start_random_h, high=self.start_random_h)
        start_h = (center_h + self.start_from_center_bias_h + start_random_bias_h)
        # start l
        start_random_bias_l = np.random.uniform(low=-self.start_random_l, high=self.start_random_l)
        start_l = (center_l + side_sign*self.start_from_center_bias_l + start_random_bias_l)
        # start w
        start_random_bias_w = np.random.uniform(low=-self.start_random_w, high=self.start_random_w)
        ws = np.where(innermask[0,int(round(start_h)),:,int(round(start_l))])
        if(len(ws[0])!=0):
            close_w = np.min(ws)
            start_w = int(round(close_w + self.start_from_close_bias_w + start_random_bias_w))
        else:
            start_w = int(round(center_w + self.start_from_center_bias_w + start_random_bias_w))
        # combine
        start_pos=np.array([start_h,start_w,start_l])
        
        # gen skeleton dst
        # dst h
        dst_random_bias_h = np.random.uniform(low=-self.dst_random_h, high=self.dst_random_h)
        dst_h = start_h + self.dst_from_start_bias_h + dst_random_bias_h
        # dst l
        dst_random_bias_l = np.random.uniform(low=-self.dst_random_l, high=self.dst_random_l)
        dst_l = start_l + side_sign*self.dst_from_start_bias_l + dst_random_bias_l
        # dst w
        dst_random_bias_w = np.random.uniform(low=-self.dst_random_w, high=self.dst_random_w)
        dst_w = start_w + self.dst_from_start_bias_w + dst_random_bias_w
        # combine
        dst_pos = np.array([dst_h,dst_w,dst_l])

        self.debug_print(f"center_h:{center_h} center_w:{center_w} center_l:{center_l}")
        self.debug_print(f"start_pos:{start_pos} dst_pos:{dst_pos}")
        return start_pos, dst_pos

    def gen_pos_random_by_start_pos(self, start_pos, side_sign):
        """
            left side: side_sign=1
            right side: side_sign=-1
        """
        start_h, start_w, start_l = start_pos
        
        # gen skeleton dst
        # dst h
        dst_random_bias_h = np.random.uniform(low=-self.dst_random_h, high=self.dst_random_h)
        dst_h = start_h + self.dst_from_start_bias_h + dst_random_bias_h
        # dst l
        dst_random_bias_l = np.random.uniform(low=-self.dst_random_l, high=self.dst_random_l)
        dst_l = start_l + side_sign*self.dst_from_start_bias_l + dst_random_bias_l
        # dst w
        dst_random_bias_w = np.random.uniform(low=-self.dst_random_w, high=self.dst_random_w)
        dst_w = start_w + self.dst_from_start_bias_w + dst_random_bias_w
        # combine
        dst_pos = np.array([dst_h,dst_w,dst_l])

        self.debug_print(f"start_pos:{start_pos} dst_pos:{dst_pos}")
        return start_pos, dst_pos
    
    def gen_skeleton_param(self, start_pos, dst_pos, step=60, eps=1e-6):
        """
        Generate skeleton param-skeleton center `step_list` and skeleton layer height and width `range_list` from 
        skeleton start position `start_pos` and dst position `dst_pos`
        
        Output:
            step_list:
                List, float32.
            range_list:
                List, [float32, float32].
        """
        start_dir = norm_vec(dst_pos-start_pos)
        init_dif_dist = cal_point_dist(point_a=start_pos, point_b=dst_pos)
        init_dif_height = dst_pos[0] - start_pos[0]
        current_dir = start_dir
        current_pos = copy.copy(start_pos)
        init_cube_size = self.init_cube_size
        max_dif_height = dst_pos[0]-start_pos[0]
        self.debug_print(f"in gen skeleton start_pos:{start_pos}\ndst_pos:{dst_pos}\nstep:{step}")
        step_list = []
        range_list = []
        for step_idx in range(0,step):
            
            cur_dif_dist = cal_point_dist(point_a=current_pos, point_b=dst_pos)
            cur_dif_height = dst_pos[0] - current_pos[0]
            cur_alpha = cur_dif_dist/init_dif_dist
            
            # cube size grow gradually
            if(self.use_gradual_grow):
                cube_size = (cur_dif_height/max_dif_height)*init_cube_size

            # collect skeleton regions
            centroid_w = current_pos[1]
            centroid_l = current_pos[2]
            min_collect_h = current_pos[0] - cube_size//2
            max_collect_h = min_collect_h + cube_size
            for collect_h in range(min_collect_h, max_collect_h):
                range_w = cube_size * 0.5
                range_l = cube_size * 0.5
                step_list.append(collect_h)
                range_list.append([centroid_w, centroid_l, range_w, range_l])
            self.debug_print(f"test current_pos:{current_pos} current_dir:{current_dir}")
            
            # check whether reach destination
            if(cal_point_dist(current_pos, dst_pos)<=eps):
                break
            
            # move forward one step
            delta_dir = norm_vec(np.random.normal(loc=0,scale=1,size=[3]))
            # delta_size = np.random.uniform(low=0, high=self.skeleton_delta_size_per_step)
            delta_size = self.skeleton_delta_size_per_step
            reachdst_dir = norm_vec(dst_pos - current_pos)
            reachdst_size = self.skeleton_reachdst_size_per_step * ((1/(cur_alpha+eps))**3)
            current_dir = norm_vec(current_dir + delta_dir*delta_size + reachdst_dir*reachdst_size)
            current_pos += current_dir
            if(cal_point_dist(current_pos, dst_pos)<1.0):
                current_pos = dst_pos
                        
        return step_list, range_list
    
    def gen_skeleton_by_param(self, step_list, range_list, reduce_w=0, reduce_l=0):
        """
        Generate skeleton_mask from skeleton center `step_list` and skeleton layer height and width `range_list`
        
        Output:
            skeleton_mask:
                array, [1, h, w, l] float32.
        """
        skeleton = np.zeros(shape=[128,128,128],dtype=np.float32)
        for step_idx in range(0, len(step_list)):
            h = step_list[step_idx]
            centroid_w, centroid_l, range_w, range_l = range_list[step_idx]
            range_w = np.clip(range_w - reduce_w, 1.0, 10000.0)
            range_l = np.clip(range_l - reduce_l, 1.0, 10000.0)
            mask_lb_pos = np.array([h, centroid_w - range_w, centroid_l - range_l])
            mask_rt_pos = np.array([h, centroid_w + range_w, centroid_l + range_l])
            # clip and tidy
            h = np.round(np.clip(h, 0, 127)).astype(np.int32)
            mask_lb_pos = np.round(np.clip(mask_lb_pos, 0, 127)).astype(np.int32)
            mask_rt_pos = np.round(np.clip(mask_rt_pos, 0, 127)).astype(np.int32)
            # print(f"gen skeleton h:{h} mask_lb_pos:{mask_lb_pos} mask_rt_pos:{mask_rt_pos}")
            # fill
            if(step_idx==0):
                last_h = h-1
            else:
                last_h = np.round(np.clip(step_list[step_idx-1], 0, 127)).astype(np.int32)
            skeleton[last_h+1:h+1, mask_lb_pos[1]:mask_rt_pos[1], mask_lb_pos[2]:mask_rt_pos[2]] = 1
        # random transform
        if(self.use_random_transform_skeleton):
            skeleton=random_transform_3D(skeleton,iter_num=1, \
                deform_limits=(self.random_transform_limit_low, self.random_transform_limit_high))
        # tidy
        skeleton = skeleton[np.newaxis,:,:,:]
        skeleton = np.where(skeleton>1e-1, 1, 0).astype(np.float32)
        return skeleton
            
    def gen_skeleton(self,start_pos, dst_pos, step=25, ret_list_flag=True):
        # output skeleton [c,h,w,l]
        step_list, range_list = self.gen_skeleton_param(
                                    start_pos = start_pos,
                                    dst_pos = dst_pos,
                                    step = step)
        skeleton = self.gen_skeleton_by_param(
                                    step_list = step_list,
                                    range_list = range_list)
        if(ret_list_flag):
            return skeleton, step_list, range_list
        return skeleton
    
    def gen_vis_skeleton_by_param(self, 
                                  step_list, 
                                  range_list,
                                  range_w=2,
                                  range_l=2):
        """
        Generate vis_skeleton_mask from skeleton center `step_list` and skeleton layer height and width `range_list`, for visualize.
        
        Output:
            skeleton_mask:
                array, [1, h, w, l] float32.
        """
        skeleton = np.zeros(shape=[128,128,128],dtype=np.float32)
        # print(f"test gen_vis_skeleton_by_param range_w:{range_w} range_l:{range_l}")
        for step_idx in range(0, len(step_list)):
            h = step_list[step_idx]
            centroid_w, centroid_l, _, _ = range_list[step_idx]
            # range_w = np.clip(range_w - reduce_w, 1.0, 10000.0)
            # range_l = np.clip(range_l - reduce_l, 1.0, 10000.0)
            mask_lb_pos = np.array([h, centroid_w - range_w/2, centroid_l - range_l/2])
            mask_rt_pos = np.array([h, centroid_w + range_w/2, centroid_l + range_l/2])
            # clip and tidy
            h = int(np.clip(h, 0, 127))
            mask_lb_pos = np.round(np.clip(mask_lb_pos, 0, 127)).astype(np.int32)
            mask_rt_pos = np.round(np.clip(mask_rt_pos, 0, 127)).astype(np.int32)
            mask_rt_pos = np.clip(mask_rt_pos, a_min=mask_lb_pos+1, a_max=None)
            # fill
            skeleton[h:h+1, mask_lb_pos[1]:mask_rt_pos[1], mask_lb_pos[2]:mask_rt_pos[2]] = 1
        # random transform
        if(self.use_random_transform_skeleton):
            skeleton=random_transform_3D(skeleton,iter_num=1, \
                deform_limits=(self.random_transform_limit_low, self.random_transform_limit_high))
        # tidy
        skeleton = skeleton[np.newaxis,:,:,:]
        skeleton = np.where(skeleton>1e-1, 1, 0).astype(np.float32)
        return skeleton

    def dot_skeleton(self, skeletonmask, innermask):
        cleftmask = skeletonmask*innermask
        cleftmask = np.where(cleftmask>1e-1, 1, 0)
        return cleftmask
    
    def dilate_skeleton(self, cleftmask, innermask, dilate_time, ret_before_dot_flag=False):
        """
        Generate dilated skeleton_mask list for init cleft_mask `cleft_mask`, maxi_mask `innermask`, and dilate iteration times 
        `dilate_time`
        
        Input:
            clefmask:
                array, [1, h, w, l] float32
            innermask:
                array, [1, h, w, l] float32
        
        Output:
            cleftmask_list:
                List, each element is array, [1, h, w, l] float32.
            defactmask_list:
                List, each element is array, [1, h, w, l] float32.
        """
        #dilate cleftmask
        cleftmask_list = []
        defactmask_list = []
        before_dot_cleftmask_list = []
        
        defactmask = (1-cleftmask)*innermask
        if(self.use_clean_chips_all):
            cleftmask, defactmask = self.clean_chip_func(cleftmask=cleftmask, defactmask=defactmask)

        cleftmask_list.append(cleftmask)
        defactmask_list.append(defactmask)
        before_dot_cleftmask_list.append(cleftmask)
        for i in range(0,dilate_time):
            self.debug_print(f"dilate time:{i} vol:{np.sum(cleftmask_list[-1])} dot:{np.sum(cleftmask_list[-1]*innermask)}")
            prev_cleftmask = copy.copy(cleftmask_list[-1])
            dilate_cleftmask = self.dilate_mask(cleftmask=cleftmask_list[-1])
            # random transform
            if((self.use_random_transform_dilate) and (i!=dilate_time-1)):
                trans_dilate_cleftmask = random_transform_4D(image=dilate_cleftmask,iter_num=1,deform_limits=(0.25,0.3))
            else:
                trans_dilate_cleftmask = dilate_cleftmask
            dilate_cleftmask = np.where(np.logical_or(trans_dilate_cleftmask>1e-1, prev_cleftmask>1e-1), 1, 0).astype(np.float32)
            before_dot_cleftmask_list.append(copy.deepcopy(dilate_cleftmask))
            dilate_cleftmask = np.where(dilate_cleftmask*innermask>0.1,1,0)
            defactmask = np.where((1-dilate_cleftmask)*innermask>0.1,1,0)
            if(self.use_clean_chips_all):
                dilate_cleftmask, defactmask = self.clean_chip_func(cleftmask = dilate_cleftmask, defactmask=defactmask)
            cleftmask_list.append(dilate_cleftmask.astype(np.float32))
            defactmask_list.append(defactmask.astype(np.float32))
        if(ret_before_dot_flag):
            return cleftmask_list, defactmask_list, before_dot_cleftmask_list
        return cleftmask_list, defactmask_list
    
    def dilate_mask(self,cleftmask):
        cleftmask=torch.FloatTensor(cleftmask[np.newaxis,:,:,:,:])
        conv_weight=get_init_kernel3()
        filtermask=F.conv3d(cleftmask,weight=conv_weight,bias=None,stride=1,padding=1)
        filtermask=F.dropout2d(filtermask,p=0.5)
        spread_cleftmask=np.where((filtermask+cleftmask.numpy())>0.1,1.0,0.0)
        return spread_cleftmask[0]
    
    def clean_chip_func(self, cleftmask, defactmask, debug=False):
        cur_cleftmask = cleftmask[0]
        cur_defactmask = defactmask[0]
        keep_list = get_component_3D(cur_defactmask, keep_num=1000)
        for com, vol in keep_list:
            self.debug_print(f"vol:{vol}")
            if(vol < 10):
                center = get_centroid(com)
                intersect_flag = (check_contour_intersect(base_mask=cur_cleftmask, mask_b=com)==True)
                self.debug_print(f"center: {center} intersect: {intersect_flag}")
                if(intersect_flag):
                    cur_cleftmask = get_union_mask(base_mask=cur_cleftmask, add_mask=com)
                    cur_defactmask = get_dif_mask(base_mask=cur_defactmask, minus_mask=com)
        cleftmask = cur_cleftmask[np.newaxis, :, :, :]
        defactmask = cur_defactmask[np.newaxis, :, :, :]
        return cleftmask, defactmask
    
    def gen_mask_core(self,
                      image,
                      innermask,
                      gen_left=True,
                      step=45,
                      dilate_time=5):
        gen_right = not gen_left
        side_sign = 1 if gen_left else -1
        # filter air hole
        innermask = np.where(image>self.thresh_air,1,0)*innermask
        # calculate skeleton center
        start_pos, dst_pos = self.gen_pos_random(innermask, side_sign)
    
        # gen step_list, range_list
        step_list, range_list = self.gen_skeleton_param(
                                    start_pos = start_pos,
                                    dst_pos = dst_pos,
                                    step = step)
        # skeleton
        skeletonmask = self.gen_skeleton_by_param(
                                    step_list = step_list,
                                    range_list = range_list)
        # vis skeleton
        vis_skeletonmask = self.gen_vis_skeleton_by_param(
                                    step_list = step_list,
                                    range_list = range_list)
        # dot skeleton
        init_cleftmask = self.dot_skeleton(skeletonmask, innermask)
        # dilate skeleton
        cleftmask_list, defactmask_list = self.dilate_skeleton(init_cleftmask, innermask, dilate_time)
        cleftmask = cleftmask_list[-1]
        defactmask = defactmask_list[-1]
        
        # clean chips
        if(self.use_clean_chips):
            cleftmask, defactmask = self.clean_chip_func(cleftmask=cleftmask, defactmask=defactmask)
            
            
        bbxmask = get_mask_bbx4D_array(cleftmask, pad_h=4, pad_w=4, pad_l=4)
        result_dict = {
            "image" : image,
            "step" : step,
            "dilate_time" : dilate_time,
            "step_list" : step_list,
            "range_list" : range_list,
            "innermask" : innermask,
            "skeletonmask" : skeletonmask,
            "vis_skeletonmask" : vis_skeletonmask,
            "init_cleftmask" : init_cleftmask,
            "cleftmask" : cleftmask,
            "defactmask" : defactmask,
            "bbxmask" : bbxmask,
            "cleftmask_list" : cleftmask_list,
            "defactmask_list" : defactmask_list
        }
        return result_dict
    
    # below are result dict functions
    def totensor_result_dict(self, result_dict):
        result_dict["image"] = totensor(result_dict["image"])
        result_dict["bbxmask"] = totensor(result_dict["bbxmask"])
        result_dict["innermask"] = totensor(result_dict["innermask"])
        result_dict["skeletonmask"] = totensor(result_dict["skeletonmask"])
        result_dict["vis_skeletonmask"] = totensor(result_dict["vis_skeletonmask"])
        result_dict["init_cleftmask"] = totensor(result_dict["init_cleftmask"])
        result_dict["cleftmask"] = totensor(result_dict["cleftmask"])
        result_dict["defactmask"] = totensor(result_dict["defactmask"])
        result_dict["cleftmask_list"] = totensor(result_dict["cleftmask_list"])
        result_dict["defactmask_list"] = totensor(result_dict["defactmask_list"])
        return result_dict
    
    def merge_result_dict(self, result_dict_left, result_dict_right):
        # image
        image = result_dict_left["image"]
        # step
        step_left = result_dict_left["step"]
        step_right = result_dict_right["step"]
        step = max(step_left, step_right)
        # dilate_time
        dilate_time_left = result_dict_left["dilate_time"]
        dilate_time_right = result_dict_right["dilate_time"]
        dilate_time = max(dilate_time_left, dilate_time_right)
        # step list
        step_list_left = result_dict_left["step_list"]
        step_list_right = result_dict_right["step_list"]
        step_list = [step_list_left, step_list_right]
        # range list
        range_list_left = result_dict_left["range_list"]
        range_list_right = result_dict_right["range_list"]
        range_list = [range_list_left, range_list_right]
        # innermask
        innermask = result_dict_left["innermask"]
        # skeletonmask
        skeletonmask_left = result_dict_left["skeletonmask"]
        skeletonmask_right = result_dict_right["skeletonmask"]
        skeletonmask = np.logical_or(skeletonmask_left, skeletonmask_right)
        # vis skeletonmask
        vis_skeletonmask_left = result_dict_left["vis_skeletonmask"]
        vis_skeletonmask_right = result_dict_right["vis_skeletonmask"]
        vis_skeletonmask = np.logical_or(vis_skeletonmask_left, vis_skeletonmask_right)
        # init cleftmask
        init_cleftmask_left = result_dict_left["init_cleftmask"]
        init_cleftmask_right = result_dict_right["init_cleftmask"]
        init_cleftmask = np.logical_or(init_cleftmask_left, init_cleftmask_right)
        # cleftmask
        cleftmask_left = result_dict_left["cleftmask"]
        cleftmask_right = result_dict_right["cleftmask"]
        cleftmask = np.logical_or(cleftmask_left, cleftmask_right)
        # defactmask
        defactmask_left = result_dict_left["defactmask"]
        defactmask_right = result_dict_right["defactmask"]
        defactmask = np.logical_and(defactmask_left, defactmask_right)
        # cleftmask_list
        cleftmask_list = []
        cleftmask_list_left = result_dict_left["cleftmask_list"]
        cleftmask_list_right = result_dict_right["cleftmask_list"]
        cleftmask_num_left = len(cleftmask_list_left)
        cleftmask_num_right = len(cleftmask_list_right)
        cleftmask_num = max(cleftmask_num_left, cleftmask_num_right)
        for cleftmask_idx in range(0, cleftmask_num):
            # left
            if(cleftmask_idx<cleftmask_num_left):
                cleftmask_left_step = cleftmask_list_left[cleftmask_idx]
            else:
                cleftmask_left_step = cleftmask_list_left[-1]
            # right
            if(cleftmask_idx<cleftmask_num_right):
                cleftmask_right_step = cleftmask_list_right[cleftmask_idx]
            else:
                cleftmask_right_step = cleftmask_list_right[-1]
            cleftmask_step = np.logical_or(cleftmask_left_step, cleftmask_right_step)
            cleftmask_list.append(cleftmask_step)
        # defactmask_list
        defactmask_list = []
        defactmask_list_left = result_dict_left["defactmask_list"]
        defactmask_list_right = result_dict_right["defactmask_list"]
        defactmask_num_left = len(defactmask_list_left)
        defactmask_num_right = len(defactmask_list_right)
        defactmask_num = max(defactmask_num_left, defactmask_num_right)
        for defactmask_idx in range(0, defactmask_num):
            # left
            if(defactmask_idx<defactmask_num_left):
                defactmask_left_step = defactmask_list_left[defactmask_idx]
            else:
                defactmask_left_step = defactmask_list_left[-1]
            # right
            if(defactmask_idx<defactmask_num_right):
                defactmask_right_step = defactmask_list_right[defactmask_idx]
            else:
                defactmask_right_step = defactmask_list_right[-1]
            defactmask_step = np.logical_and(defactmask_left_step, defactmask_right_step)
            defactmask_list.append(defactmask_step)
            
        bbxmask = get_mask_bbx4D_array(cleftmask, pad_h=4, pad_w=4, pad_l=4)
        # result_dict
        merge_result_dict = {
            "image" : image,
            "step_left" : step_left,
            "step_right" : step_right,
            "step" : step,
            "dilate_time_left" : dilate_time_left,
            "dilate_time_right" : dilate_time_right,
            "dilate_time" : dilate_time,
            "step_list_left" : step_list_left,
            "step_list_right" : step_list_right,
            "step_list" : step_list,
            "range_list_left" : range_list_left,
            "range_list_right" : range_list_right,
            "range_list" : range_list,
            "innermask" : innermask,
            "skeletonmask" : skeletonmask,
            "vis_skeletonmask" : vis_skeletonmask,
            "init_cleftmask" : init_cleftmask,
            "cleftmask" : cleftmask,
            "defactmask" : defactmask,
            "bbxmask" :  bbxmask,
            "cleftmask_list" : cleftmask_list,
            "defactmask_list" : defactmask_list
        }
        return merge_result_dict

    def get_random_dilate_time(self):
        return np.random.randint(low=self.dilate_random_low, high=self.dilate_random_high+1)
    
    def gen_mask_random_dict(self,
                        image,
                        innermask=None,
                        gen_unilateral=None,
                        dilate_time=None,
                        gen_syn_image=False
                        ):
        '''
        input:
            image [c,h,w,l] 4D tensor
            mask [c,h,w,l] 4D tensor
            gen_unilateral [True or False]
        output:
            result_dict{mask,cleftmask,defactmask [c,h,w,l] tensor}
        '''
        if(innermask is None):
            pre_mask=torch.ones_like(image)
            innermask=gen_determine_innermask(image,pre_mask,self.thresh_bone)
        image, innermask = tonumpy([image,innermask])

        if(gen_unilateral is None):
            gen_unilateral = np.random.uniform(low=0,high=1)<self.gen_unilateral_possibility
          
        if(gen_unilateral):
            # one side
            rand_dice = np.random.uniform(0,1)
            gen_left = rand_dice<=0.5
            if(dilate_time is None):
                dilate_time = np.random.randint(low=self.dilate_random_low, high=self.dilate_random_high+1)
            step = np.random.randint(low=self.step_random_low, high=self.step_random_high)
            result_dict = self.gen_mask_core(image, innermask, gen_left=gen_left, step=step, dilate_time=dilate_time)
        else:
            # left
            if(dilate_time is None):
                dilate_time_left = np.random.randint(low=self.dilate_random_low, high=self.dilate_random_high+1)
            else:
                dilate_time_left = dilate_time
            step_left = np.random.randint(low=self.step_random_low, high=self.step_random_high)
            result_dict_left = self.gen_mask_core(image, innermask, gen_left=True, step=step_left, dilate_time=dilate_time_left)
            # right
            if(dilate_time is None):
                dilate_time_right = np.random.randint(low=self.dilate_random_low, high=self.dilate_random_high+1)
            else:
                dilate_time_right = dilate_time
            step_right = np.random.randint(low=self.step_random_low, high=self.step_random_high)
            result_dict_right = self.gen_mask_core(image,innermask,gen_left=False,step=step_right,dilate_time=dilate_time_right)
            # merge
            result_dict = self.merge_result_dict(result_dict_left, result_dict_right)
        
        result_dict = self.totensor_result_dict(result_dict)
        # print(f"test result_dict.keys:{result_dict.keys()}")
        if(gen_syn_image):
            result_dict["syn_image"] = None
        return result_dict
    
    def gen_mask_random(self,
                        image,
                        innermask=None,
                        gen_unilateral=None,
                        gen_syn_image=False):
        '''
        input:
            image [c,h,w,l] tensor
        
        output:
            bbxmask, cleftmask, defactmask [c,h,w,l] tensor
        '''
        result_dict = self.gen_mask_random_dict(
            image = image,
            innermask = innermask,
            gen_unilateral = gen_unilateral,
            gen_syn_image = gen_syn_image
        )
        bbxmask = result_dict["bbxmask"]
        cleftmask = result_dict["cleftmask"]
        defactmask = result_dict["defactmask"] * bbxmask
        return totensor([bbxmask, cleftmask, defactmask])
    
    def gen_mask_random_with_centeralign(self,
                        image,
                        innermask,
                        ref_center,
                        gen_unilateral=None,
                        gen_syn_image=False,
                        ret_trans_result_dict=False,
                        debug=False
                        ):
        ref_center_t = torch.FloatTensor(ref_center)
        cur_center_t = get_centroid_tensor4D(innermask)
        trans_h, trans_w, trans_l = ref_center_t - cur_center_t
        self.debug_print(f"trans_h:{trans_h} trans_w:{trans_w} trans_l:{trans_l}")
        trans_image = spatial_transform_by_param_tensor4D(
            image = image,
            trans_x = trans_h,
            trans_y = trans_w,
            trans_z = trans_l,
            mode = "bilinear"
        )
        trans_innermask = spatial_transform_by_param_tensor4D(
            image = innermask,
            trans_x = trans_h,
            trans_y = trans_w,
            trans_z = trans_l,
            mode = "nearest"
        )
        trans_result_dict = self.gen_mask_random_dict(
            image = trans_image,
            innermask = trans_innermask,
            gen_unilateral = gen_unilateral,
            gen_syn_image = gen_syn_image
        )

        trans_result_dict["trans_image"] = trans_image
        trans_result_dict["trans_innermask"] = trans_innermask

        trans_bbxmask = trans_result_dict["bbxmask"]
        trans_cleftmask = trans_result_dict["cleftmask"]
        trans_defactmask = trans_result_dict["defactmask"] * trans_bbxmask
        
        restore_bbxmask = spatial_transform_by_param_tensor4D(
            image = trans_bbxmask,
            trans_x = -trans_h,
            trans_y = -trans_w,
            trans_z = -trans_l,
            mode = "nearest"
        )
        restore_cleftmask = spatial_transform_by_param_tensor4D(
            image = trans_cleftmask,
            trans_x = -trans_h,
            trans_y = -trans_w,
            trans_z = -trans_l,
            mode = "nearest"
        )
        restore_defactmask = spatial_transform_by_param_tensor4D(
            image = trans_defactmask,
            trans_x = -trans_h,
            trans_y = -trans_w,
            trans_z = -trans_l,
            mode = "nearest"
        )
        if(debug):
            trans_skeletonmask = trans_result_dict["skeletonmask"]
            restore_skeletonmask = spatial_transform_by_param_tensor4D(
            image = trans_skeletonmask,
            trans_x = -trans_h,
            trans_y = -trans_w,
            trans_z = -trans_l,
            mode = "nearest"
            )
            debug_result_dict = {
                    "bbxmask" : totensor(restore_bbxmask),
                    "cleftmask" : totensor(restore_cleftmask),
                    "defactmask" : totensor(restore_defactmask),
                    "skeletonmask" : totensor(restore_skeletonmask),
                    "trans_innermask" : totensor(trans_innermask)
                    }
            return debug_result_dict
        
        if(ret_trans_result_dict):
            return trans_result_dict
        return totensor([restore_bbxmask, restore_cleftmask, restore_defactmask])
    
    
    # below are batch generate functions
    def gen_indicatemask_batch(self,image_batch,gen_unilateral_p=0.7):
        innermask_batch=[None]*image_batch.shape[0]
        return self.gen_mask_random_batch(image_batch,innermask_batch,gen_unilateral_p)
    
    def gen_mask_random_batch(self,image_batch,innermask_batch,gen_unilateral_p=0.7):
        #input image [b,c,h,w,l] tensor
        #output mask_batch, innermask_batch, imitatemask_batch
        #use absolute location, default assum the cube is of shape 128*128*128
        origin_device=image_batch.device
        batch_num=image_batch.shape[0]
        mask_list,cleftmask_list,defactmask_list=[],[],[]
        for batch_idx in range(0,batch_num):
            mask,cleftmask,defactmask=self.gen_mask_random(image_batch[batch_idx],innermask_batch[batch_idx],gen_unilateral_p)
            mask_list.append(mask.to(origin_device))
            cleftmask_list.append(cleftmask.to(origin_device))
            defactmask_list.append(defactmask.to(origin_device))
        return torch.stack(mask_list),torch.stack(cleftmask_list),torch.stack(defactmask_list)