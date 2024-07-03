class ModelManager(object):
    def __init__(self,model_list):
        self.model_list=model_list
        
    def cuda(self):
        for model in self.model_list:
            model=model.cuda()
    
    def train(self):
        for model in self.model_list:
            model.train()
        
    def eval(self):
        for model in self.model_list:
            model.eval()
        