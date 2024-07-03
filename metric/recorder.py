import numpy as np
import logging
import matplotlib.pyplot as plt

class BasicRecorder:
    def __init__(self, name,  root_dir, init_value=0):
        self.name = name
        self.root_dir = root_dir
        self.value = init_value
        self.count = 0
        
        # init logger
        self.logger_name = self.name.replace("/","_")
        self.log_path = f"{self.root_dir}/{self.logger_name}.txt"        
        logger = logging.getLogger(self.logger_name)
        log_file_handler = logging.FileHandler(filename=self.log_path,mode="a")
        logger.addHandler(log_file_handler)
        logger.setLevel(logging.INFO)
        
    def update(self, update_value):
        if(type(update_value) is not np.ndarray and type(update_value) is not float):
            detach_update_value = update_value.detach().cpu().item()
        else:
            detach_update_value = update_value
        self.value += detach_update_value
        self.count += 1
    
    def get_report_value(self):
        report_value = self.gen_report_value(self.value, self.count)
        return report_value
    
    def report(self):
        report_value = self.gen_report_value(self.value, self.count)
        report_msg = self.gen_report_msg(report_value)
        log_logger = logging.getLogger(self.logger_name)
        log_logger.info(report_value)
        self.value=0
        self.count=0
        return report_msg

    def gen_report_value(self, value, count):
        if(value==0):
            report_value = 0
        else:
            report_value = value/count
        return report_value

    def gen_report_msg(self, report_value):
        msg=f"{self.name}: {report_value}"
        return msg

    def get_value_list(self):
        log_file = open(self.log_path, mode="r")
        log_lines = log_file.readlines()
        log_lines = [log_line.replace("\n","") for log_line in log_lines]
        value_list = [float(log_line) for log_line in log_lines]
        return value_list

    def draw(self):
        draw_path = f"{self.root_dir}/{self.logger_name}.png"
        value_list = self.get_value_list()
        y = np.array(value_list).astype(np.float32)
        x = np.arange(0,len(y)).astype(np.int32)
        plt.plot(x,y, label=self.logger_name)
        plt.grid()
        plt.title(self.name)
        plt.legend()
        plt.savefig(draw_path)
        plt.close()
        return value_list
