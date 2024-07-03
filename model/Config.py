import os
import torch
import torch.nn as nn
from .common import log

class Ckpt(nn.Module):
    def __init__(self,step=0,epoch=0,learning_rate=1e-4):
        super().__init__()
        self.config_dict={}
        self.config_dict["step"]=step
        self.config_dict["epoch"]=epoch
        self.config_dict["learning_rate"]=learning_rate
    
    def __getitem__(self, key):
        return self.config_dict[key]
    
    def __setitem__(self, key, value):
        self.config_dict[key]=value
    
    def __delitem__(self, key):
        self.config_dict.__delitem__(key)
    
    def load(self,config_path):
        log(f"loading Config: {config_path}")
        if(os.path.exists(config_path)):
            self.config_dict=torch.load(config_path)
            log(f"Config loaded.")
        else:
            log(f"Config path doesn't exist, config created.")
        
    def save(self,config_path):
        log(f"saving config:{config_path}")
        torch.save(self.config_dict,config_path)
        log("config saved.")
