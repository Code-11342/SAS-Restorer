import sys
sys.path.append("../")
from dependencies import *
from utils import *

class EvalRealDataset3D(Dataset):
    def __init__(self,
                 data_dir,
                 thresh_bone=1500/2300,
                 thresh_air=800/2300,
                 ret_name=True,
                 mask_dir_name="mask",
                 maxi_dir_name="maxi"):
        super().__init__()
        self.thresh_air = thresh_air
        self.thresh_bone = thresh_bone
        self.data_dir = data_dir
        self.ret_name = ret_name
        self.mask_dir_name = mask_dir_name
        self.maxi_dir_name = maxi_dir_name
        self.image_paths=glob.glob(data_dir)
        print(f"using EvalRealDataset3D with {len(self.image_paths)} images")
        print(f"mask_dir_name: {self.mask_dir_name}")
        print(f"maxi_dir_name: {self.maxi_dir_name}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        image_path=self.image_paths[idx]
        name = get_name(image_path)
        # image
        image = read_mha_tensor4D(image_path)
        # mask
        mask_path = image_path.replace("image", f"{self.mask_dir_name}")
        mask = read_mha_tensor4D(mask_path)
        # bony_border_mask
        bony_border_mask = torch.where(image>self.thresh_bone, 1, 0).float()*mask

        image, mask, bony_border_mask = tocuda([image,mask,bony_border_mask])
        if(self.ret_name):
            return image, mask, bony_border_mask, name
        return image, mask, bony_border_mask