import os

import numpy as np
import scipy as scp
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

from random import seed
import random
from numpy import random as np_random
from scipy.stats import spearmanr, entropy

import gc

from captum import attr as captum_attr

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################################################################################
# Wrappers
################################################################################################ 
    
class wrapper_class_mask(nn.Module):
    def __init__(self,model,is_softmax=False):
        super(wrapper_class_mask, self).__init__()
        self.model=model
        self.is_softmax = is_softmax
    def forward(self, x, target, model_additional):
        default_device = x.get_device()
        logits = self.model(x,model_additional)
        pred_labels = torch.argmax(logits, dim=-1).squeeze()
        if self.is_softmax:
            logits[:] = F.softmax(logits, dim=-1)
        
        # masking
        for i in range(self.model.n_classes):
            mask_target = torch.full(pred_labels.shape, False).to(default_device)
            mask_target[torch.where(pred_labels==i)]=True
            logits[:,:,:,i] = logits[:,:,:,i]*mask_target
            
        del mask_target, pred_labels
        gc.collect()
        torch.cuda.empty_cache()
        return logits.sum(dim=(1,2)) # output shape=(1,C)
    
class wrapper_logit(nn.Module):
    def __init__(self,model,is_softmax=False):
        super(wrapper_logit, self).__init__()
        self.model=model
        self.is_softmax = is_softmax
    def forward(self, x, target, model_additional):
        logits = self.model(x,model_additional)
        if self.is_softmax:
            logits[:] = F.softmax(logits, dim=-1)
        return logits.sum(dim=(1,2)) # output shape=(1,C)

    
class wrapper_class(nn.Module):
    def __init__(self,model,is_softmax=False):
        super(wrapper_class, self).__init__()
        self.model=model
        self.is_softmax = is_softmax
    def forward(self, x, target, model_additional):
        # additional = target
        logits = self.model(x,model_additional)
        if self.is_softmax:
            logits[:] = F.softmax(logits, dim=-1)
        for i in range(self.model.n_classes):
            if i != target:
                logits[:,:,:,i] *= 0
        return logits.sum(dim=(1,2)) # output shape=(1,C)
    
class wrapper_gt(nn.Module):
    def __init__(self,model,is_softmax=False):
        super(wrapper_gt, self).__init__()
        self.model=model
        self.is_softmax = is_softmax
    def forward(self, x, gt, target, model_additional):
        # additional = gt
        default_device = x.get_device()
        logits = self.model(x,model_additional)
        if self.is_softmax:
            logits[:] = F.softmax(logits, dim=-1)
        
        # masking
        for i in range(self.model.n_classes):
            logits[:,i,:,:] = logits[:,i,:,:]*(gt==target)            
        return logits.sum(dim=(1,2)) # output shape=(1,C)
    

class wrapper_region_class(nn.Module):
    def __init__(self,model,is_softmax=False):
        super(wrapper_region_class, self).__init__()
        self.model=model
        self.is_softmax = is_softmax
    def forward(self, x, target, region, model_additional):
        # additional = target
        logits = self.model(x,model_additional)
        if self.is_softmax:
            logits[:] = F.softmax(logits, dim=-1)
        b = x.shape[0]
        for i in range(b):
            for c in range(self.model.n_classes):
                logits[i,:,:,c] *= region[i]       
        for i in range(self.model.n_classes):
            if i != target:
                logits[:,:,:,i] *= 0
        
        # Logit shape = batch x HW x C
        return logits.sum(dim=(1,2)) # output shape=(1,C)

    
class wrapper_region(nn.Module):
    def __init__(self,model,is_softmax=False):
        super(wrapper_region, self).__init__()
        self.model=model
        self.is_softmax = is_softmax
    def forward(self, x, target, region, model_additional):
        # additional = target
        logits = self.model(x,model_additional)
        if self.is_softmax:
            logits[:] = F.softmax(logits, dim=-1) 
        b = x.shape[0]        
        for i in range(b):
            for c in range(self.model.n_classes):
                logits[i,:,:,c][~region[i]] *= 0
        
        # Logit shape = batch x HW x C
        return logits.sum(dim=(1,2)) # output shape=(1,C)    
    

class wrapper_region_class_mask(nn.Module):
    def __init__(self,model,is_softmax=False):
        super(wrapper_class_mask, self).__init__()
        self.model=model
        self.is_softmax = is_softmax
    def forward(self, x, target, region, model_additional):
        default_device = x.get_device()
        logits = self.model(x,model_additional)
        pred_labels = torch.argmax(logits, dim=-1).squeeze()
        if self.is_softmax:
            logits[:] = F.softmax(logits, dim=-1)
        
        # masking
        for i in range(self.model.n_classes):
            mask_target = torch.full(pred_labels.shape, 0).to(default_device)
            mask_target[torch.where(pred_labels==i)]=1
            logits[:,:,:,i] = logits[:,:,:,i]*mask_target

        b = x.shape[0]        
        for i in range(b):
            for c in range(self.model.n_classes):
                logits[i,:,:,c][~region[i]] *= 0
            
        del mask_target, pred_labels
        gc.collect()
        torch.cuda.empty_cache()
        return logits.sum(dim=(1,2)) # output shape=(1,C)
    
    
def select_wrapper(model, wrapper_type, is_softmax=False, device='cpu'):
    if wrapper_type == 'class_mask':
        wrapper_model = wrapper_class_mask(model,is_softmax)
    elif wrapper_type == 'logit':
        wrapper_model = wrapper_logit(model,is_softmax)
    elif wrapper_type == 'class':
        wrapper_model = wrapper_class(model,is_softmax)    
    elif wrapper_type == 'gt':
        wrapper_model = wrapper_gt(model,is_softmax) 
    elif wrapper_type == 'region':
        wrapper_model = wrapper_region(model,is_softmax)         
    elif wrapper_type == 'region_class':
        wrapper_model = wrapper_region_class(model,is_softmax)             
    elif wrapper_type == 'region_class_mask':
        wrapper_model = wrapper_region_class(model,is_softmax)            
    wrapper_model = wrapper_model.to(device)
    wrapper_model.eval()
    
    return wrapper_model

def select_additional(wrapper_type, target, gt, region, model_additional):
    if wrapper_type == 'class_mask':
        additional = (0,model_additional)
    elif wrapper_type == 'logit':
        additional = (0,model_additional)
    elif wrapper_type == 'class':
        additional =(target,model_additional)
    elif wrapper_type == 'gt':
        additional = (gt,target,model_additional)
    elif wrapper_type == 'region':
        additional = (target,region,model_additional)
    elif wrapper_type == 'region_class':
        additional = (target,region,model_additional) 
    elif wrapper_type == 'region_class_mask':
        additional = (target,region,model_additional)         
    
    return additional

################################################################################################
# Attribution
################################################################################################

def attribution_generator(attribution_method, attribution_model):
    # Set up attribution
    attribution_model.eval()
    
    if attribution_method=='GuidedBackprop':
        attr = captum_attr.GuidedBackprop(attribution_model)
    elif attribution_method=='Saliency':
        attr = captum_attr.Saliency(attribution_model)
    elif attribution_method=='InputXGradient':
        attr = captum_attr.InputXGradient(attribution_model) 
    elif attribution_method=='IntegratedGradients':
        attr = captum_attr.IntegratedGradients(attribution_model, multiply_by_inputs=False) 
    elif attribution_method=='GuidedGradCam':
        attr = captum_attr.GuidedGradCam(attribution_model, attribution_model.model.up4.conv)
    elif attribution_method=='GradCam':
        attr = captum_attr.GuidedGradCam(attribution_model, attribution_model.model.outc.conv)        
    elif attribution_method=='LRP':
        # Need to use specialized model for LRP
        attr = captum_attr.LRP(attribution_model) 
    else:
        raise NameError(f"Attribution method {attribution_method} is not a valid method.")
    
    return attr

def baseline_generator(method, input_x, baseline_dict):    
    input_shape = input_x.shape
    input_step = input_shape[1]
    input_batch = input_x.shape[0]
    if method=='local_min':
        baseline = np.min(input_x)*np.ones(input_shape)
    elif method=='local_max':
        baseline = np.max(input_x)*np.ones(input_shape)     
    elif method=='local_avg':
        baseline = np.mean(input_x)*np.ones(input_shape)
    elif method=='local_minmax':
        baseline = (np.min(input_x)+np.max(input_x))/2*np.ones(input_shape)        
    elif method=='stepwise_min':
        baseline = np.min(np.min(input_x,axis=2),axis=2)
        baseline = np.expand_dims(np.expand_dims(baseline,axis=2),axis=2)
        baseline = np.tile(baseline,(1,1,input_shape[2],input_shape[3]))
    elif method=='stepwise_max':
        baseline = np.max(np.max(input_x,axis=2),axis=2)
        baseline = np.expand_dims(np.expand_dims(baseline,axis=2),axis=2)
        baseline = np.tile(baseline,(1,1,input_shape[2],input_shape[3]))
    elif method=='stepwise_avg':
        baseline = np.mean(np.mean(input_x,axis=2),axis=2)
        baseline = np.expand_dims(np.expand_dims(baseline,axis=2),axis=2)
        baseline = np.tile(baseline,(1,1,input_shape[2],input_shape[3]))
    elif method=='stepwise_minmax':
        baseline_min = np.min(np.min(input_x,axis=2),axis=2)
        baseline_max = np.max(np.max(input_x,axis=2),axis=2)
        baseline = np.expand_dims(np.expand_dims((baseline_min+baseline_max)/2,axis=2),axis=2)
        baseline = np.tile(baseline,(1,1,input_shape[2],input_shape[3]))        
    elif method=='pixelwise_min':
        baseline = np.expand_dims(baseline_dict['pixelwise_min'],axis=0)
        baseline = np.tile(baseline,(input_batch,1,1,1))        
    elif method=='pixelwise_max':
        baseline = np.expand_dims(baseline_dict['pixelwise_max'],axis=0)
        baseline = np.tile(baseline,(input_batch,1,1,1))           
    elif method=='pixelwise_avg':
        baseline = np.expand_dims(baseline_dict['pixelwise_avg'],axis=0)
        baseline = np.tile(baseline,(input_batch,1,1,1))              
    elif method=='pixelwise_minmax':
        baseline = np.expand_dims(baseline_dict['pixelwise_minmax'],axis=0)
        baseline = np.tile(baseline,(input_batch,1,1,1))                    
    elif method=='channelwise_min':
        baseline = np.expand_dims(baseline_dict['channelwise_min'],axis=0)
        baseline = np.tile(baseline,(input_batch,1,1,1))           
    elif method=='channelwise_max':
        baseline = np.expand_dims(baseline_dict['channelwise_max'],axis=0)
        baseline = np.tile(baseline,(input_batch,1,1,1))           
    elif method=='channelwise_avg':
        baseline = np.expand_dims(baseline_dict['channelwise_avg'],axis=0)
        baseline = np.tile(baseline,(input_batch,1,1,1))                
    elif method=='channelwise_minmax':
        baseline = np.expand_dims(baseline_dict['channelwise_minmax'],axis=0)
        baseline = np.tile(baseline,(input_batch,1,1,1))            
    elif method=='first_channel':
        baseline = input_x[:,0:1,:,:]
        baseline = np.tile(baseline,(1,input_step,1,1))
    elif method=='last_channel':
        baseline = input_x[:,-1:,:,:]
        baseline = np.tile(baseline,(1,input_step,1,1))        
    elif method=='zero':
        baseline = np.zeros(input_shape)        
    else:
        raise NameError(f"Attribution baseline {method} is not a valid baseline.")
    
    return baseline

def perform_attribution(attribution_method, 
                        attr,
                        input_x,
                        baseline,
                        additional,
                        target):
    b_size = input_x.shape[0]
    if attribution_method=='GuidedBackprop':
            attribution = attr.attribute(input_x, target=target, 
                                         additional_forward_args=additional)
    elif attribution_method=='Saliency':
        attribution = attr.attribute(inputs=input_x, target=target, abs=False,
                                    additional_forward_args=additional)
    elif attribution_method=='Deconvolution':
        attribution = attr.attribute(input_x, target=target, 
                                     additional_forward_args=additional)
    elif attribution_method=='InputXGradient':
        attribution = attr.attribute(input_x, target=target,
                                    additional_forward_args=additional)
    elif attribution_method=='IntegratedGradients':
        attribution = attr.attribute(input_x, target=target, 
                               baselines=baseline,
                               n_steps=25, internal_batch_size = b_size,
                               return_convergence_delta=False,
                               additional_forward_args=additional, 
                               )
    elif attribution_method=='GuidedGradCam':
        attribution = attr.attribute(input_x,target=target,interpolate_mode='nearest', attribute_to_layer_input=False,
                                    additional_forward_args=additional)
    elif attribution_method=='GradCam':
        attribution = attr.attribute(input_x,target=target,interpolate_mode='nearest', attribute_to_layer_input=False,
                                    additional_forward_args=additional)          
    elif attribution_method=='LRP':
        attribution = attr.attribute(inputs=input_x-baseline, target=target,
                                  return_convergence_delta=False,
                                  additional_forward_args=additional,
                                    )

    return attribution.detach().cpu()

def attributor(net, data_loader, wrapper_type, is_softmax, method, baseline_type, target, baseline_dict, device='cpu'):

    wrapper_model = select_wrapper(net, wrapper_type, is_softmax)
    attr_method = attribution_generator(method, wrapper_model)    
    wrapper_model.eval()
    
    attribution_result_list = []
    
    for inputs, gt in tqdm(iter(data_loader)):
        inputs_gpu = inputs.to(device)
        gt_gpu = gt.to(device)
        baseline = torch.Tensor(baseline_generator(baseline_type, inputs.numpy(), baseline_dict)).to(device)
        additional = select_additional(wrapper_type, target, gt_gpu)
        
        attribution_result = perform_attribution(method, 
                                                attr_method,
                                                inputs_gpu,
                                                baseline,
                                                additional,
                                                target)
        
        attribution_result_list.append(attribution_result.numpy())

        del inputs_gpu, gt_gpu, baseline
        gc.collect()
        torch.cuda.empty_cache()

    return attribution_result_list


