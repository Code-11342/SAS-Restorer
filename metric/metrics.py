import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from .recorder import BasicRecorder

class AvgMetric:
    def __init__(self,name="default_name",root_dir="./log_dir",init_value=0,train_flag=True,is_metric=False,new_line=False):
        self.name = name
        self.root_name = name.replace("/","_")
        self.root_dir = f"{root_dir}/{self.root_name}"
        os.makedirs(self.root_dir, exist_ok=True)
        self.train_flag = train_flag
        self.is_metric_flag = is_metric
        self.new_line = new_line
        # log
        self.log_recorder = BasicRecorder(name=f"{self.name}_log", root_dir=self.root_dir, init_value=init_value)
        # train
        self.train_recorder = BasicRecorder(name=f"{self.name}_train", root_dir=self.root_dir, init_value=init_value)
        # val
        self.val_recorder = BasicRecorder(name=f"{self.name}_val", root_dir=self.root_dir, init_value=init_value)
    
    # status change functions
    def train(self):
        self.train_flag=True
    
    def eval(self):
        self.train_flag=False
    
    # update
    def update(self,update_value):
        self.log_recorder.update(update_value)
        if(self.train_flag):
            self.train_recorder.update(update_value)
        else:
            self.val_recorder.update(update_value)
    
    # report functions
    def report_log(self):
        return self.log_recorder.report()
    
    def report_train(self):
        return self.train_recorder.report()
    
    def report_val(self):
        return self.val_recorder.report()
    
    # get value functions
    def get_report_value_log(self):
        return self.log_recorder.get_report_value()
    
    def get_report_value_train(self):
        return self.train_recorder.get_report_value()

    def get_report_value_val(self):
        return self.val_recorder.get_report_value()
    
    # value list functions
    def get_log_value_list(self):
        return self.log_recorder.get_value_list()
    
    def get_train_value_list(self):
        return self.train_recorder.get_value_list()

    def get_val_value_list(self):
        return self.val_recorder.get_value_list()

    # draw functions
    def draw_log(self):
        return self.log_recorder.draw()
    
    def draw_train(self):
        return self.train_recorder.draw()
    
    def draw_val(self):
        return self.val_recorder.draw()
    
    # information functions
    def get_name(self):
        return self.name
    
    # metric
    def is_metric(self):
        return self.is_metric_flag