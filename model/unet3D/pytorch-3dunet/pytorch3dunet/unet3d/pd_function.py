import torch
import importlib
import numpy as np

class VanillaPredictFunction:
    def __init__(self):
        pass
    
    def __call__(self, input):
        input = input[:,:,:,:,:]
        return input

class ThreshPredictFunction:
    def __init__(self, pd_function_config):
        self.thresh = pd_function_config["thresh"]
        print("Initialize ThreshPredictFunction")
        print(f"thresh: {self.thresh}")
    
    def __call__(self, input):
        if(isinstance(input, np.ndarray)):
            return np.where(input>self.thresh, 1, 0)
        else:
            return torch.where(input>self.thresh, 1, 0)

class ArgmaxPredictFunction:
    def __init__(self, pd_function_config):
        print("Initialize ArgmaxPredictFunction")
    
    def __call__(self, input):
        if(isinstance(input, np.ndarray)):
            return np.argmax(input, axis=1)[:, np.newaxis, :, :, :]
        else:
            return torch.argmax(input, dim=1, keepdim=True)

def get_pd_function(config):
    def _pd_function_class(class_name):
        m = importlib.import_module('pytorch3dunet.unet3d.pd_function')
        clazz = getattr(m, class_name)
        return clazz
    if("pd_function" not in config):
        print("Could not find pd function configuration, using VanillaPredictFunction")
        return VanillaPredictFunction()
    else:
        pd_function_config = config["pd_function"]
        pd_function_class = _pd_function_class(pd_function_config["name"])
        return pd_function_class(pd_function_config)