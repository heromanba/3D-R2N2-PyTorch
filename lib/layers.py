#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from lib.config import cfg

import collections

import torch
import torch.nn as nn
from torch.autograd import Variable


###############################################################################
#                                                                             #
#                FCConv3DLayer using PyTorch                                  #
#                                                                             #
###############################################################################
class FCConv3DLayer_torch(nn.Module):
    def __init__(self, fc_w_fan_in, filter_shape, output_shape):
        print("initializing \"FCConv3DLayer_torch\"")
        super(FCConv3DLayer_torch, self).__init__()
        self.output_shape = output_shape
        
        #fc_layer is not the same as fc7
        self.fc_layer = nn.Linear(fc_w_fan_in, int(np.prod(output_shape[1:])), bias=False)
        
        #filter_shape = (in_channels, out_channels, kernel_d, kernel_h, kernel_w)
        self.conv3d = nn.Conv3d(filter_shape[0], filter_shape[1], \
                                kernel_size= filter_shape[2], \
                                padding= int((filter_shape[2] - 1) / 2), bias=False)
        
        #define a bias term and initialize it to 0.1
        self.bias = nn.Parameter(torch.FloatTensor(1, output_shape[1], 1, 1, 1).fill_(0.1))
    
    def forward(self, fc7, h):
        #fc7 is the leakyReLU-ed ouput of fc7 layer
        #h is the hidden state of the previous time step
        target_shape = list(self.output_shape)

        # To deal with different batch_size.
        target_shape[0] = -1    

        out = self.fc_layer(fc7).view(*target_shape) + self.conv3d(h) + self.bias
        return out
    
class BN_FCConv3DLayer_torch(nn.Module):
    def __init__(self, fc_w_fan_in, filter_shape, output_shape):
        print("initializing \"FCConv3DLayer_torch\"")
        super(BN_FCConv3DLayer_torch, self).__init__()
        self.output_shape = output_shape
        
        #fc_layer is not the same as fc7
        self.fc_layer = nn.Linear(fc_w_fan_in, int(np.prod(output_shape[1:])), bias=False)
        
        #filter_shape = (in_channels, out_channels, kernel_d, kernel_h, kernel_w)
        self.conv3d = nn.Conv3d(filter_shape[0], filter_shape[1], \
                                kernel_size= filter_shape[2], \
                                padding= int((filter_shape[2] - 1) / 2), bias=False)
        
        #define the recurrent batch normalization layers
        #input channels is the output channels of FCConv3DLayer_torch and T_max is the maximum number of views
        self.bn1 = Recurrent_BatchNorm3d(num_features = filter_shape[0], T_max = cfg.CONST.N_VIEWS)
        self.bn2 = Recurrent_BatchNorm3d(num_features = filter_shape[0], T_max = cfg.CONST.N_VIEWS)
        
        #define a bias term and initialize it to 0.1
        self.bias = nn.Parameter(torch.FloatTensor(1, output_shape[1], 1, 1, 1).fill_(0.1))
    
    def forward(self, fc7, h, time):
        #fc7 is the leakyReLU-ed ouput of fc7 layer
        #h is the hidden state of the previous time step
        target_shape = list(self.output_shape)

        # To deal with different batch_size.
        target_shape[0] = -1    

        fc7 = self.fc_layer(fc7).view(*target_shape)
        bn_fc7 = self.bn1(fc7, time)    #the input of Recurrent_BatchNorm3d is (input_, time)
        
        conv3d = self.conv3d(h) 
        bn_conv3d = self.bn2(conv3d, time)
        
        out = bn_fc7 + bn_conv3d + self.bias
        return out
    
###############################################################################
#                                                                             #
#                unpooling layer using PyTorch                                #
#                                                                             #
###############################################################################
class Unpool3DLayer(nn.Module):
    def __init__(self, unpool_size=2, padding=0):
        print("initializing \"Unpool3DLayer\"")
        super(Unpool3DLayer, self).__init__()
        self.unpool_size = unpool_size
        self.padding = padding
    
    def forward(self, x):
        n = self.unpool_size
        p = self.padding
        #x.size() is (batch_size, channels, depth, height, width)
        output_size = (x.size(0), x.size(1), n * x.size(2), n * x.size(3), n * x.size(4))
        
        out_tensor = torch.Tensor(*output_size).zero_()
        
        if torch.cuda.is_available():
            out_tensor = out_tensor.cuda()
            
        out = out_tensor

        out[:, \
            :, \
            p : p + output_size[2] + 1 : n, \
            p : p + output_size[3] + 1 : n, \
            p : p + output_size[4] + 1 : n] = x
        return out
    
###############################################################################
#                                                                             #
#                        softmax with loss 3D                                 #
#                                                                             #
###############################################################################
class SoftmaxWithLoss3D(nn.Module):
    def __init__(self):
        print("initializing \"SoftmaxWithLoss3D\"")
        super(SoftmaxWithLoss3D, self).__init__()
    
    def forward(self, inputs, y=None, test=False):
        
        if type(test) is not bool:
            raise Exception("keyword argument \"test\" needs to be a bool type")
        if (test == False) and (y is None):
            raise Exception("\"y is None\" and \"test is False\" cannot happen at the same time")
            
        """
        Before actually compute the loss, we need to address the possible numberical instability.
        If some elements of inputs are very large, and we compute their exponential value, then we
        might encounter some infinity. So we need to subtract them by the largest value along the 
        "channels" dimension to avoid very large exponential.
        """
        #the size of inputs and y is (batch_size, channels, depth, height, width)
        #torch.max return a tuple of (max_value, index_of_max_value)
        max_channel = torch.max(inputs, dim = 1, keepdim = True)[0]
        adj_inputs = inputs - max_channel
        
        exp_x = torch.exp(adj_inputs)
        sum_exp_x = torch.sum(exp_x, dim = 1, keepdim = True)
        
        #if the ground truth is provided the loss will be computed
        if y is not None:
            loss = torch.mean(
                              torch.sum(-y * adj_inputs, dim = 1, keepdim = True) + \
                              torch.log(sum_exp_x))
            
        #if this is in the test mode, then the prediction and loss need to be returned
        if test:
            prediction = exp_x / sum_exp_x
            if y is not None:
                return [prediction, loss]
            else:
                return [prediction]
        return loss


###############################################################################
#                                                                             #
#                Recurrent batch normalization using PyTorch                  #
#                                                                             #
###############################################################################
class Recurrent_BatchNorm3d(nn.Module):
    #use similar APIs as torch.nn.BatchNorm3d.
    def __init__(self, \
                 num_features, \
                 T_max, \
                 eps=1e-5, \
                 momentum=0.1, \
                 affine=True, \
                 track_running_stats=True):
        super(Recurrent_BatchNorm3d, self).__init__()
        #num_features is C from an expected input of size (N, C, D, H, W)
        self.num_features = num_features
        self.T_max = T_max
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats=True
        
        #if affine is true, this module has learnable affine parameters
        #weight is gamma and bias is beta in the batch normalization formula
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        #if track_running_stats is True, this module will track the running mean and running variance
        for i in range(T_max):
            self.register_buffer('running_mean_{}'.format(i), \
                                 torch.zeros(num_features) if track_running_stats else None)
            self.register_buffer('running_var_{}'.format(i), \
                                 torch.zeros(num_features) if track_running_stats else None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.track_running_stats:
            for i in range(self.T_max):
                running_mean = getattr(self, 'running_mean_{}'.format(i))
                running_var = getattr(self, 'running_var_{}'.format(i))
                
                running_mean.zero_()
                running_var.fill_(1)
        
        if self.affine:
            #according to the paper, 0.1 is a good initialization for gamma
            self.weight.data.fill_(0.1)
            self.bias.data.zero_()
            
    def _check_input_dim(self, input_):
            if input_.dim() != 5:
                raise ValueError('expected 5D input (got {}D input)'.format(input_.dim()))
    
    def forward(self, input_, time):
        self._check_input_dim(input_)
        if time >= self.T_max:
            time = self.T_max - 1
            
        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))
        
        return nn.functional.batch_norm(input_, \
                                        running_mean = running_mean, \
                                        running_var = running_var, \
                                        weight = self.weight, \
                                        bias = self.bias, \
                                        training = self.training, \
                                        momentum = self.momentum, \
                                        eps = self.eps)
        
    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' T_max={T_max}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


if __name__ == '__main__':
    recur_batchnorm = Recurrent_BatchNorm3d(3, 3)
