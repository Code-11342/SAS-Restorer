import sys
sys.path.append("../")
from .dependencies import *
from utils import *

class CompleteDataset3DFixMask(Dataset):
    def __init__(self, mha_dir, ret_name=False):
        super().__init__()
        self.thresh_air = 800/2300
        self.thresh_bone = 1500/2300
        self.mha_dir = mha_dir
        self.ret_name = ret_name
        self.mha_paths = glob.glob(self.mha_dir)
        print(f"Using CompleteDataset3DFixMask")
        print(f"dataset size:{len(self.mha_paths)}")
    
    def __len__(self):
        return len(self.mha_paths)
    
    def __getitem__(self,idx):
        mha_path = self.mha_paths[idx]
        name = get_name(mha_path)
        # image
        image_path = mha_path
        x = read_mha_tensor4D(image_path)
        # bbx_mask
        bbx_mask_path = mha_path.replace("gt_image", "mask")
        bbx_mask = read_mha_tensor4D(bbx_mask_path)
        mask = bbx_mask
        # bony_border_mask
        bony_border_mask = torch.where(x>self.thresh_bone, 1, 0).float()*mask

        x, mask, bony_border_mask = tocuda([x, mask, bony_border_mask])
        if(self.ret_name):
            return x, mask, bony_border_mask, name
        return x, mask, bony_border_mask