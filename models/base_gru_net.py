#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:04:40 2018

@author: wangchu
"""


from models.net import Net
from lib.layers import SoftmaxWithLoss3D

import torch
from torch.autograd import Variable
##########################################################################################
#                                                                                        #
#                      GRUNet definition using PyTorch                                   #
#                                                                                        #
##########################################################################################
class BaseGRUNet(Net):
    """
    This class is used to define some common attributes and methods that both GRUNet and 
    ResidualGRUNet have. Note that GRUNet and ResidualGRUNet have the same loss function
    and forward pass. The only difference is different encoder and decoder architecture.
    """
    def __init__(self):
        print("initializing \"BaseGRUNet\"")
        super(BaseGRUNet, self).__init__()
        """
        Set the necessary data of the network
        """
        self.is_x_tensor4 = False
        
        self.n_gru_vox = 4
        #the size of x is (num_views, batch_size, 3, img_w, img_h)
        self.input_shape = (self.batch_size, 3, self.img_w, self.img_h)
        #number of filters for each convolution layer in the encoder
        self.n_convfilter = [96, 128, 256, 256, 256, 256]
        #the dimension of the fully connected layer
        self.n_fc_filters = [1024]
        #number of filters for each 3d convolution layer in the decoder
        self.n_deconvfilter = [128, 128, 128, 64, 32, 2]
        #the size of the hidden state
        self.h_shape = (self.batch_size, self.n_deconvfilter[0], self.n_gru_vox, self.n_gru_vox, self.n_gru_vox)
        #the filter shape of the 3d convolutional gru unit
        self.conv3d_filter_shape = (self.n_deconvfilter[0], self.n_deconvfilter[0], 3, 3, 3)
        
        #set the last layer 
        self.SoftmaxWithLoss3D = SoftmaxWithLoss3D()
        
        
        #set the encoder and the decoder of the network
        self.encoder = None
        self.decoder = None
        
    def forward(self, x, y=None, test=True):
        #ensure that the network has encoder and decoder attributes
        if self.encoder is None:
            raise Exception("subclass network of BaseGRUNet must define the \"encoder\" attribute")
        if self.decoder is None:
            raise Exception("subclass network of BaseGRUNet must define the \"decoder\" attribute")

        #initialize the hidden state and update gate
        h = self.initHidden(self.h_shape)
        u = self.initHidden(self.h_shape)
        
        #a list used to store intermediate update gate activations
        u_list = []
        
        """
        x is the input and the size of x is (num_views, batch_size, channels, heights, widths).
        h and u is the hidden state and activation of last time step respectively.
        The following loop computes the forward pass of the whole network. 
        """
        for time in range(x.size(0)):
            gru_out, update_gate = self.encoder(x[time], h, u, time)
            
            h = gru_out
            
            u = update_gate
            u_list.append(u)
        
        out = self.decoder(h)
        
        """
        If test is True and y is None, then the out is the [prediction].
        If test is True and y is not None, then the out is [prediction, loss].
        If test is False and y is not None, then the out is loss.
        """
        out = self.SoftmaxWithLoss3D(out, y=y, test=test)
        if test:
            out.extend(u_list)
        return out
    
    def initHidden(self, h_shape):
        h = torch.zeros(h_shape)
        if torch.cuda.is_available():
            h = h.cuda()
        return Variable(h)
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        