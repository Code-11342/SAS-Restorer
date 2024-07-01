import numpy as np
import matplotlib.pyplot as plt
from .metrics import AvgMetric

class MetricManager:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.metric_groups={}
        self.metric_names=[]
        self.train_flag=True
        self.log_interval=None
        self.train_interval=None
        self.val_interval=None
    
    def get_metric_name(self, metric_name, perfix=""):
        origin_perfix=perfix
        if(perfix!=""):
            perfix=perfix+"_"
        #field_name and last_name
        field_name=""
        last_name=metric_name
        if("/"in metric_name):
            split_idx=metric_name.rindex("/")+1
            field_name=metric_name[:split_idx]
            last_name=metric_name[split_idx:]
        #metric_name
        metric_name=field_name+perfix+last_name
        #group_name
        group_name=field_name+"_"+origin_perfix
        return metric_name, group_name
    
    def update(self,metric_name,metric_value,perfix="", is_metric=False, new_line=False):
        metric_name, group_name = self.get_metric_name(metric_name=metric_name, perfix=perfix)
        #add into group
        if(group_name not in self.metric_groups):
            self.metric_groups[group_name]={}
        metric_group=self.metric_groups[group_name]
        if(metric_name not in metric_group):
            metric_group[metric_name]=AvgMetric(name=metric_name,root_dir=self.root_dir,train_flag=self.train_flag,is_metric=is_metric, new_line=new_line)
        self.metric_groups[group_name][metric_name].update(metric_value)
    
    def get_metric_value_log(self, metric_name, perfix=""):
        metric_name, group_name = self.get_metric_name(metric_name=metric_name, perfix=perfix)
        return self.metric_groups[group_name][metric_name].get_report_value_log()
    
    def get_metric_value_train(self, metric_name, perfix=""):
        metric_name, group_name = self.get_metric_name(metric_name=metric_name, perfix=perfix)
        return self.metric_groups[group_name][metric_name].get_report_value_train()
    
    def get_metric_value_val(self, metric_name, perfix=""):
        metric_name, group_name = self.get_metric_name(metric_name=metric_name, perfix=perfix)
        return self.metric_groups[group_name][metric_name].get_report_value_val()
    
    def train(self):
        self.train_flag=True
        for group_name in self.metric_groups.keys():
            metric_group=self.metric_groups[group_name]
            for metric_name in metric_group.keys():
                metric=metric_group[metric_name]
                metric.train()
        return self
    
    def eval(self):
        self.train_flag=True
        for group_name in self.metric_groups.keys():
            metric_group=self.metric_groups[group_name]
            for metric_name in metric_group.keys():
                metric=metric_group[metric_name]
                metric.eval()
        return self
    
    def set_interval(self,log_interval,train_interval,val_interval):
        self.log_interval=log_interval
        self.train_interval=train_interval
        self.val_interval=val_interval
        for group_name in self.metric_groups.keys():
            metric_group=self.metric_groups[group_name]
            for metric_name in metric_group.keys():
                metric=metric_group[metric_name]
                metric.log_interval=log_interval
                metric.train_interval=train_interval
                metric.val_interval=val_interval
        return self
    
    # report functions
    def report_log(self):
        msg=""
        for group_name in self.metric_groups.keys():
            metric_group=self.metric_groups[group_name]
            for midx,metric_name in enumerate(metric_group.keys()):
                metric=metric_group[metric_name]
                if(midx%3==0 and midx!=0):
                    msg+="\n"
                msg+=metric.report_log()+" "
            msg+="\n"
        msg=msg.replace("\n\n","\n")
        return msg
    
    def report_train(self):
        msg=""
        for group_name in self.metric_groups.keys():
            metric_group=self.metric_groups[group_name]
            for midx,metric_name in enumerate(metric_group.keys()):
                metric=metric_group[metric_name]
                if(midx%3==0 and midx!=0):
                    msg+="\n"
                msg+=metric.report_train()+" "
            msg+="\n"
        msg=msg.replace("\n\n","\n")
        return msg
    
    def report_val(self):
        msg=""
        for group_name in self.metric_groups.keys():
            metric_group=self.metric_groups[group_name]
            for midx,metric_name in enumerate(metric_group.keys()):
                metric=metric_group[metric_name]
                if(midx%3==0 and midx!=0):
                    msg+="\n"
                msg+=metric.report_val()+" "
            msg+="\n"
        msg=msg.replace("\n\n","\n")
        return msg

    # draw functions
    def draw_log(self, draw_all=True):
        all_log_value_list = []
        all_log_name_list = []
        for group_name in self.metric_groups.keys():
            metric_group=self.metric_groups[group_name]
            for midx,metric_name in enumerate(metric_group.keys()):
                metric=metric_group[metric_name]
                log_value_list = metric.draw_log()
                if(not metric.is_metric()):
                    all_log_value_list.append(log_value_list)
                    all_log_name_list.append(metric.get_name())
        if(draw_all):
            plt.figure(dpi=250)
            all_log_plot_path = f"{self.root_dir}/all_log.png"
            for log_name,log_value_list in zip(all_log_name_list,all_log_value_list):
                y = np.array(log_value_list).astype(np.float32)
                x = np.arange(0,len(y)).astype(np.int32)
                plt.plot(x,y, label=log_name)
            plt.legend()
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.title("log loss")
            plt.savefig(all_log_plot_path)
            plt.close()
    
    def draw_train(self, draw_all=True):
        all_train_value_list = []
        all_train_name_list = []
        for group_name in self.metric_groups.keys():
            metric_group=self.metric_groups[group_name]
            for midx,metric_name in enumerate(metric_group.keys()):
                metric=metric_group[metric_name]
                train_value_list = metric.draw_train()
                if(not metric.is_metric()):
                    all_train_value_list.append(train_value_list)
                    all_train_name_list.append(metric.get_name())
        if(draw_all):
            plt.figure(dpi=250)
            all_train_plot_path = f"{self.root_dir}/all_train.png"
            for train_name,train_value_list in zip(all_train_name_list,all_train_value_list):
                y = np.array(train_value_list).astype(np.float32)
                x = np.arange(0,len(y)).astype(np.int32)
                plt.plot(x,y, label=train_name)
            plt.legend()
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.title("train loss")
            plt.savefig(all_train_plot_path)
            plt.close()
    
    def draw_val(self, draw_all=True):
        all_val_value_list = []
        all_val_name_list = []
        for group_name in self.metric_groups.keys():
            metric_group=self.metric_groups[group_name]
            for midx,metric_name in enumerate(metric_group.keys()):
                metric=metric_group[metric_name]
                val_value_list = metric.draw_val()
                if(not metric.is_metric()):
                    all_val_value_list.append(val_value_list)
                    all_val_name_list.append(metric.get_name())
        if(draw_all):
            plt.figure(dpi=250)
            all_val_plot_path = f"{self.root_dir}/all_val.png"
            for val_name,val_value_list in zip(all_val_name_list,all_val_value_list):
                y = np.array(val_value_list).astype(np.float32)
                x = np.arange(0,len(y)).astype(np.int32)
                plt.plot(x, y, label=val_name)
            plt.legend()
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.title("val loss")
            plt.savefig(all_val_plot_path)
            plt.close()

    def draw(self, draw_log=True, draw_train=True, draw_val=True):
        if(draw_log):
            self.draw_log(draw_all=True)
        if(draw_train):
            self.draw_train(draw_all=True)
        if(draw_val):
            self.draw_val(draw_all=True)