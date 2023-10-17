import numpy as np
import scipy as scp
import pandas as pd

import torch
from torch import nn

# Use smaller size
class RadarAE(nn.Module):
    def __init__(self, in_channels, normF=nn.BatchNorm2d):
        super(RadarAE, self).__init__()
        self.encoder = nn.Sequential(
                        nn.Conv2d(in_channels,out_channels = 16, kernel_size = 3,padding=1, bias=False), # 22                        
                        nn.ReLU(),
                        normF(16),
                        nn.MaxPool2d(2,2),                          # 20                                    
                        nn.Conv2d(16,16,3,padding=1, bias=False),   # 10                        
                        nn.ReLU(),
                        normF(16),
                        nn.Conv2d(16,32,3,padding=1, bias=False),   # 8                                                 
                        nn.ReLU(),
                        normF(32),
                        nn.MaxPool2d(2,2),                          # 6                                     
                        nn.Conv2d(32,32,3,padding=1, bias=False),   # 3                       
                        nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(32,32,3,stride = 2,padding = 1, output_padding = 1, bias=False),            
                        nn.ReLU(),
                        normF(32),
                        nn.ConvTranspose2d(32,16,3,1,1, bias=False),                                                
                        nn.ReLU(),
                        normF(16),
                        nn.ConvTranspose2d(in_channels=16,out_channels=in_channels,kernel_size = 3, stride = 2,padding = 1, output_padding = 1, bias=False)                     
        )

                
    def forward(self,x):
        encoded_img = self.encoder(x)
        reconstructed_img = self.decoder(encoded_img)
        return reconstructed_img

# Reduce channels at each layer 
class RadarAE2(nn.Module):
    def __init__(self, in_channels):
        super(RadarAE2, self).__init__()
        self.l1 = in_channels-1
        self.l2 = in_channels-2
        self.l3 = in_channels-3
        self.l4 = in_channels-4
        self.encoder1 = nn.Sequential(
                        nn.Conv2d(in_channels,out_channels = self.l1, kernel_size = 3,padding=1, bias=False), # 22                 
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),                          # 20                                    
                        nn.Conv2d(self.l1,self.l2,3,padding=1, bias=False),   # 10                        
                        nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(    
                        nn.Conv2d(self.l2,self.l3,3,padding=1, bias=False),   # 8                                                 
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),                          # 6                                     
                        nn.Conv2d(self.l3,self.l4,3,padding=1, bias=False),   # 3                       
                        nn.ReLU()
        )
        
        self.decoder1 = nn.Sequential(
                        nn.ConvTranspose2d(self.l4,self.l2,3,stride = 2,padding = 1, output_padding = 1, bias=False),            
                        nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
                        nn.ConvTranspose2d(self.l2,self.l1,3,1,1, bias=False),                                                
                        nn.ReLU(),
                        nn.ConvTranspose2d(in_channels=self.l1,out_channels=in_channels,kernel_size = 3, stride = 2,padding = 1, output_padding = 1, bias=False)                     
        )

                
    def forward(self,x):
        h1 = self.encoder1(x)
        encoded_img = self.encoder2(h1)
        h2 = self.decoder1(encoded_img)
        reconstructed_img = self.decoder2(h2)
        return reconstructed_img, h1, h2
    
    
# Use even smaller size
class ExplanationAE(nn.Module):
    def __init__(self, in_channels, normF=nn.BatchNorm2d):
        super(RadarAE, self).__init__()
        self.encoder = nn.Sequential(
                        nn.Conv2d(in_channels,out_channels = 16, kernel_size = 3,padding=1, bias=False),                         
                        nn.ReLU(),
                        normF(16),
                        nn.MaxPool2d(2,2),                                                              
                        nn.Conv2d(16,16,3,padding=1, bias=False),                           
                        nn.ReLU(),
                        normF(16),
                        nn.Conv2d(16,16,3,padding=1, bias=False),                                                    
                        nn.ReLU(),
                        normF(16),
                        nn.MaxPool2d(2,2),                                                               
                        nn.Conv2d(16,16,3,padding=1, bias=False),                          
                        nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(16,16,3,stride = 2,padding = 1, output_padding = 1, bias=False),            
                        nn.ReLU(),
                        normF(16),
                        nn.ConvTranspose2d(16,16,3,1,1, bias=False),                                                
                        nn.ReLU(),
                        normF(16),
                        nn.ConvTranspose2d(in_channels=16,out_channels=in_channels,kernel_size = 3, stride = 2,padding = 1, output_padding = 1, bias=False)                     
        )

                
    def forward(self,x):
        encoded_img = self.encoder(x)
        reconstructed_img = self.decoder(encoded_img)
        return reconstructed_img
    
    
# Create an AE for each lead time
# class ExplanationAE(nn.Module):
#     def __init__(self, in_channels, normF=nn.BatchNorm2d, n_lead = 6):
#         super(ExplanationAE, self).__init__()
#         self.models = nn.ModuleList()
#         for i in range(0,n_lead):
#             self.models.append(RadarAE(in_channels, normF))
                
#     def forward(self,x, lead):
#         b = x.shape[0]
#         reconstructed = []
#         for i in range(0,b):
#             encoded = self.models[lead[i]].encoder(x[i:i+1])
#             reconstructed.append(self.models[lead[i]].decoder(encoded))
#         reconstructed_img = torch.stack(reconstructed)
#         return reconstructed_img

def radar_loss_fn(net, inputs, device,):
    inputs_gpu = inputs.to(device)
    pred = net(inputs_gpu)
    loss = torch.sum(torch.sum((inputs_gpu-pred)**2,dim=[1,2,3]))
    
    del inputs_gpu, pred    
    return loss

def radar_extract_fn(net):    
    return net.state_dict().copy()     

def radar2_loss_fn(net, inputs, device,):
    inputs_gpu = inputs.to(device)
    pred, h1, h2 = net(inputs_gpu)
    loss = torch.sum(torch.sum((inputs_gpu-pred)**2,dim=[1,2,3])) + torch.sum(torch.sum((h1-h2)**2,dim=[1,2,3]))
    
    del inputs_gpu, pred    
    return loss

def radar2_extract_fn(net):    
    return net.state_dict().copy()  

# Train attribution
def feature_att_loss_fn(net, inputs, device, **kwargs):
    inputs_gpu = inputs[0].to(device)
    
    baseline = baseline_generator(kwargs['method'], inputs_gpu, kwargs['baseline_dict'])
    attr_val = perform_attribution(kwargs['method'], kwargs['attr_fn'],inputs_gpu,baseline,kwargs['additional'],kwargs['target'])
    
    if kwargs['use_pos_only'] > 0:
        attr_val[attr_val<0] = 0
    elif kwargs['use_pos_only'] < 0:
        attr_val[attr_val>0] = 0
    
    pred = net(attr_val)
    loss = torch.sum(torch.sum((attr_val-pred)**2,dim=[1,2,3]))
    
    del inputs_gpu, pred, attr_val  
    return loss

def feature_att_extract_fn(net, **kwargs):    
    return net.state_dict().copy()     
