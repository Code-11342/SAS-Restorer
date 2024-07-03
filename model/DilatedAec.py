import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(False),
            nn.Conv3d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(False),
            nn.Conv3d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(Encoder, self).__init__()
        self.in_channels=in_channels
        self.num_hiddens=num_hiddens
        self.num_residual_layers=num_residual_layers
        self.num_residual_hiddens=num_residual_hiddens
        self.embedding_dim = embedding_dim

        #128
        self.conv_1_1 = nn.Conv3d(in_channels=in_channels,
                                 out_channels=self.num_hiddens // 8,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self.conv_1_2 = nn.Conv3d(in_channels=self.num_hiddens // 8,
                                 out_channels=self.num_hiddens // 4,
                                 kernel_size=3,
                                 stride=2, padding=2, dilation=2)
        #64
        self.conv_2_1 = nn.Conv3d(in_channels=self.num_hiddens // 4,
                                 out_channels=self.num_hiddens // 4,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self.conv_2_2 = nn.Conv3d(in_channels=self.num_hiddens // 4,
                                 out_channels=self.num_hiddens // 2,
                                 kernel_size=3,
                                 stride=2, padding=2,dilation=2)

        #32
        self.conv_3_1 = nn.Conv3d(in_channels=self.num_hiddens // 2,
                                 out_channels=self.num_hiddens // 2,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self.conv_3_2 = nn.Conv3d(in_channels=self.num_hiddens // 2,
                                 out_channels=self.num_hiddens,
                                 kernel_size=3,
                                 stride=2, padding=2,dilation=2)

        #16
        self.conv_4 = nn.Conv3d(in_channels=self.num_hiddens,
                                 out_channels=self.num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=2,dilation=2)
        #16
        self.residual_stack = ResidualStack(in_channels=self.num_hiddens,
                                             num_hiddens=self.num_hiddens,
                                             num_residual_layers=self.num_residual_layers,
                                             num_residual_hiddens=self.num_residual_hiddens)
        
        #16
        self.conv_5= nn.Conv3d(in_channels=self.num_hiddens, 
                                out_channels=self.embedding_dim,
                                kernel_size=3,
                                stride=1,padding=2,dilation=2)

    def forward(self, x, extract_feature=False):
        x = self.conv_1_1(x)
        x = F.relu(x)
        x = self.conv_1_2(x)
        x = F.relu(x)
        
        x = self.conv_2_1(x)
        x = F.relu(x)
        x = self.conv_2_2(x)
        x = F.relu(x)

        x = self.conv_3_1(x)
        x = F.relu(x)
        x = self.conv_3_2(x)
        x = F.relu(x)

        x = self.conv_4(x)
        x = F.relu(x)

        x = self.residual_stack(x)
        
        x = self.conv_5(x)
        x=F.relu(x)
        
        return x
    
    def set_requires_grad(self,flag=False):
        for param in self.parameters():
            param.requires_grad=flag

class Decoder(nn.Module):
    def __init__(self, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(Decoder, self).__init__()
        self.out_channels=out_channels
        self.num_hiddens=num_hiddens
        self.num_residual_layers=num_residual_layers
        self.num_residual_hiddens=num_residual_hiddens
        self.embedding_dim = embedding_dim

        #16
        self.conv_trans_1 = nn.ConvTranspose3d(in_channels=self.embedding_dim,
                                 out_channels=self.embedding_dim,
                                 kernel_size=3,
                                 stride=1, padding=2,dilation=2)

        #16
        self.residual_stack = ResidualStack(in_channels=self.num_hiddens,
                                             num_hiddens=self.num_hiddens,
                                             num_residual_layers=self.num_residual_layers,
                                             num_residual_hiddens=self.num_residual_hiddens)

        #16
        self.conv_trans_1_1 = nn.ConvTranspose3d(in_channels=self.embedding_dim,
                                                out_channels=self.num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        #32
        self.conv_trans_2_1 = nn.ConvTranspose3d(in_channels=self.num_hiddens // 2,
                                                out_channels=self.num_hiddens // 4,
                                                kernel_size=4,
                                                stride=2, padding=1)
        
        #64
        self.conv_trans_3_1 = nn.ConvTranspose3d(in_channels=self.num_hiddens // 4,
                                                out_channels=1,
                                                kernel_size=4,
                                                stride=2, padding=1)
        

    def forward(self, encoding):
        x = self.conv_trans_1(encoding)
        x = F.relu(x)

        x = self.residual_stack(x)

        x = self.conv_trans_1_1(x)
        x = F.relu(x)

        x = self.conv_trans_2_1(x)
        x = F.relu(x)

        out = self.conv_trans_3_1(x)

        return out
    
    def set_requires_grad(self,flag=False):
        for param in self.parameters():
            param.requires_grad=flag

def gen_innermask(image,mask,thresh=1500/2300):
    innermask=torch.where(image>thresh,1.0,0.0)*mask
    return innermask

def gen_innermask_batch(image_batch,mask_batch,thresh=(1500+1000)/2300):
    batch_num=image_batch.shape[0]
    innermask_list=[]
    for batch_idx in range(0,batch_num):
        innermask=gen_innermask(image_batch[batch_idx],mask_batch[batch_idx],thresh=thresh)
        innermask_list.append(innermask)
    return torch.stack(innermask_list)

def gen_determine_innermask(image,mask,thresh=1500/2300):
    #input image, mask [c,h,w,l] tensor
    #output innermask [c,h,w,l] tensor
    innermask=gen_innermask(image,mask,thresh=thresh)
    return innermask

def gen_determine_innermask_batch(image_batch,mask_batch,thresh=1500/2300):
    batch_num = image_batch.shape[0]
    innermask_list=[]
    for bacth_idx in range(0,batch_num):
        innermask=gen_determine_innermask(image_batch[bacth_idx],mask_batch[bacth_idx],thresh=thresh)
        innermask_list.append(innermask)
    return torch.stack(innermask_list)

class DilatedAutoEncoder(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens,embedding_dim):
        super(DilatedAutoEncoder,self).__init__()
        print(f"using DilatedAutoEncoder!")
        self.in_channels=in_channels
        self.num_hiddens=num_hiddens
        self.num_residual_layers=num_residual_layers
        self.num_residual_hiddens=num_residual_hiddens
        self.embedding_dim=embedding_dim
        self.encoder=Encoder(in_channels=self.in_channels,num_hiddens=self.num_hiddens,num_residual_layers=self.num_residual_layers,\
            num_residual_hiddens=self.num_residual_hiddens,embedding_dim=self.embedding_dim)
        self.decoder=Decoder(out_channels=self.in_channels,num_hiddens=self.num_hiddens,num_residual_layers=self.num_residual_layers,\
            num_residual_hiddens=self.num_residual_hiddens,embedding_dim=self.embedding_dim)

    def set_requires_grad(self,flag=False):
        for param in self.parameters():
            param.requires_grad=flag

    def forward(self,x):
        encoding=self.encoder.forward(x)
        rec_x=self.decoder.forward(encoding)
        return rec_x
    
    def cal_loss(self,x,rec_x):
        loss=F.mse_loss(input=rec_x,target=x)
        return loss
    
class Gendis(nn.Module):
    def __init__(self, in_channels, num_hiddens,h=96,w=96,l=96):
        super(Gendis, self).__init__()
        self.num_hiddens=num_hiddens
        self.h=h
        self.w=w
        self.l=l

        #96
        self._conv_1 = nn.Conv3d(in_channels=in_channels,
                                 out_channels=num_hiddens // 4,
                                 kernel_size=4,
                                 stride=2, padding=1)
        #48
        self._conv_2 = nn.Conv3d(in_channels=num_hiddens // 4,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        #24
        self._conv_3 = nn.Conv3d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        #12
        self._conv_4 = nn.Conv3d(in_channels=num_hiddens //2,
                                 out_channels=num_hiddens //2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        #6
        self._conv_5 = nn.Conv3d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        
        #3
        self.lin1=nn.Linear(in_features=3*3*3*num_hiddens,out_features=1024)
        self.lin2=nn.Linear(in_features=1024,out_features=1024)
        self.lin3=nn.Linear(in_features=1024,out_features=1)

    def forward(self, x):
        x=x[:,:,16:112,16:112,16:112]
        x = self._conv_1(x)
        x = F.leaky_relu(x,negative_slope=1e-2)

        x = self._conv_2(x)
        x = F.leaky_relu(x,negative_slope=1e-2)

        x = self._conv_3(x)
        x = F.leaky_relu(x,negative_slope=1e-2)

        x = self._conv_4(x)
        x=F.leaky_relu(x,negative_slope=1e-2)

        x=self._conv_5(x)
        x=F.leaky_relu(x,negative_slope=1e-2)
        
        #linears
        x=x.view(-1,3*3*3*self.num_hiddens)
        x=self.lin1(x)
        x=F.leaky_relu(x,negative_slope=1e-2)
        x=self.lin2(x)
        x=F.leaky_relu(x,negative_slope=1e-2)
        x=self.lin3(x)
        return x
    
    def set_requires_grad(self,flag=False):
        for param in self.parameters():
            param.requires_grad=flag
    
    def cal_loss(self,
                 real_x,
                 fake_x,
                 mask=None,
                 metric_manager=None,
                 eps=1e-6,
                 perfix=""):
        device = real_x.device
        wgan_gp_lambda = 10.0
        # real
        real_pd = self.forward(real_x)
        real_loss = -real_pd.mean()
        # fake
        fake_pd = self.forward(fake_x)
        fake_loss = fake_pd.mean()
        # mask
        if(mask is None):
            mask = torch.ones_like(fake_x)
        # gradient panelty
        batch_size = real_x.shape[0]
        inter_alpha = torch.rand([batch_size,1,1,1,1]).to(device)
        inter_x = autograd.Variable(inter_alpha*real_x.data+(1-inter_alpha)*fake_x.data, requires_grad=True).to(device)
        inter_pd = self.forward(inter_x)
        inter_grad = autograd.grad(outputs=inter_pd,
                                   inputs=inter_x,
                                   grad_outputs=torch.ones_like(inter_pd),
                                   create_graph=True,
                                   retain_graph=True,
                                   only_inputs=True)[0]
        
        slopes = torch.sqrt(torch.sum(torch.square(inter_grad*mask),dim=[1,2,3,4])+eps)
        wgan_gp_loss = wgan_gp_lambda*torch.mean(torch.square(slopes-1.0))
        # metric
        metric_manager.update("global_dis/real_loss", real_loss, perfix=perfix)
        metric_manager.update("global_dis/fake_loss", fake_loss, perfix=perfix)
        metric_manager.update("global_dis/gp_loss", wgan_gp_loss, perfix=perfix)
        g_total_loss = fake_loss+real_loss+wgan_gp_loss
        return g_total_loss