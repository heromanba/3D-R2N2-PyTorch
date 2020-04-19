import numpy as np

from models.base_gru_net import BaseGRUNet
from lib.layers import FCConv3DLayer_torch, Unpool3DLayer, SoftmaxWithLoss3D

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Linear, Conv2d, MaxPool2d, \
                     LeakyReLU, Conv3d, Tanh, Sigmoid

##########################################################################################
#                                                                                        #
#                      GRUNet definition using PyTorch                                   #
#                                                                                        #
##########################################################################################
class GRUNet(BaseGRUNet):
    def __init__(self):
        print("initializing \"GRUNet\"")
        super(GRUNet, self).__init__()
        """
        Set the necessary data of the network
        """
        
        #set the encoder and the decoder of the network
        self.encoder = encoder(self.input_shape, self.n_convfilter, \
                               self.n_fc_filters, self.h_shape, self.conv3d_filter_shape)
        
        self.decoder = decoder(self.n_deconvfilter, self.h_shape)
        
        #initialize all the parameters
        self.parameter_init()
        
##########################################################################################
#                                                                                        #
#                      encoder definition using PyTorch                                  #
#                                                                                        #
##########################################################################################
class encoder(nn.Module):
    def __init__(self, input_shape, n_convfilter, \
                 n_fc_filters, h_shape, conv3d_filter_shape):
        print("initializing \"encoder\"")
        #input_shape = (self.batch_size, 3, img_w, img_h)
        super(encoder, self).__init__()
        #conv1
        conv1_kernal_size = 7
        self.conv1 = Conv2d(in_channels= input_shape[1], \
                            out_channels= n_convfilter[0], \
                            kernel_size= conv1_kernal_size, \
                            padding = int((conv1_kernal_size - 1) / 2))
        
        #conv2
        conv2_kernal_size = 3
        self.conv2 = Conv2d(in_channels= n_convfilter[0], \
                            out_channels= n_convfilter[1], \
                            kernel_size= conv2_kernal_size,\
                            padding = int((conv2_kernal_size - 1) / 2))
        
        #conv3
        conv3_kernal_size = 3
        self.conv3 = Conv2d(in_channels= n_convfilter[1], \
                            out_channels= n_convfilter[2], \
                            kernel_size= conv2_kernal_size,\
                            padding = int((conv3_kernal_size - 1) / 2))
        
        #conv4
        conv4_kernal_size = 3
        self.conv4 = Conv2d(in_channels= n_convfilter[2], \
                            out_channels= n_convfilter[3], \
                            kernel_size= conv2_kernal_size,\
                            padding = int((conv4_kernal_size - 1) / 2))
        
        #conv5
        conv5_kernal_size = 3
        self.conv5 = Conv2d(in_channels= n_convfilter[3], \
                            out_channels= n_convfilter[4], \
                            kernel_size= conv2_kernal_size,\
                            padding = int((conv5_kernal_size - 1) / 2))
        
        #conv6
        conv6_kernal_size = 3
        self.conv6 = Conv2d(in_channels= n_convfilter[4], \
                            out_channels= n_convfilter[5], \
                            kernel_size= conv2_kernal_size,\
                            padding = int((conv6_kernal_size - 1) / 2))
        
        
        #pooling layer
        self.pool = MaxPool2d(kernel_size= 2, padding= 1)
        
        
        #nonlinearities of the network
        self.leaky_relu = LeakyReLU(negative_slope= 0.01)
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        
        
        #find the input feature map size of the fully connected layer
        fc7_feat_w, fc7_feat_h = self.fc_in_featmap_size(input_shape, num_pooling=6)
        #define the fully connected layer
        self.fc7 = Linear(int(n_convfilter[5] * fc7_feat_w * fc7_feat_h), n_fc_filters[0])
        
        
        #define the FCConv3DLayers in 3d convolutional gru unit
        self.t_x_s_update = FCConv3DLayer_torch(n_fc_filters[0], conv3d_filter_shape, h_shape)
        self.t_x_s_reset = FCConv3DLayer_torch(n_fc_filters[0], conv3d_filter_shape, h_shape)
        self.t_x_rs = FCConv3DLayer_torch(n_fc_filters[0], conv3d_filter_shape, h_shape)
        
    def forward(self, x, h, u):
        """
        x is the input and the size of x is (batch_size, channels, heights, widths).
        h and u is the hidden state and activation of last time step respectively.
        This function defines the forward pass of the encoder of the network.
        """
        input_to_rect6 = nn.Sequential(self.conv1, self.pool, self.leaky_relu, \
                                         self.conv2, self.pool, self.leaky_relu, \
                                         self.conv3, self.pool, self.leaky_relu, \
                                         self.conv4, self.pool, self.leaky_relu, \
                                         self.conv5, self.pool, self.leaky_relu, \
                                         self.conv6, self.pool, self.leaky_relu)
        rect6 = input_to_rect6(x)
        rect6 = rect6.view(rect6.size(0), -1)
        
        
        fc7 = self.fc7(rect6)
        rect7 = self.leaky_relu(fc7)
        
        t_x_s_update = self.t_x_s_update(rect7, h)
        t_x_s_reset = self.t_x_s_reset(rect7, h)
        
        update_gate = self.sigmoid(t_x_s_update)
        complement_update_gate = 1 - update_gate
        reset_gate = self.sigmoid(t_x_s_reset)
        
        rs = reset_gate * h
        t_x_rs = self.t_x_rs(rect7, rs)
        tanh_t_x_rs = self.tanh(t_x_rs)
        
        gru_out = update_gate * h + complement_update_gate * tanh_t_x_rs
        
        return gru_out, update_gate
        
    #infer the input feature map size, (height, width) of the fully connected layer
    def fc_in_featmap_size(self, input_shape, num_pooling):
        #fully connected layer
        img_w = input_shape[2]
        img_h = input_shape[3]
        #infer the size of the input feature map of the fully connected layer
        fc7_feat_w = img_w
        fc7_feat_h = img_h
        for i in range(num_pooling):
            #image downsampled by pooling layers
            #w_out= np.floor((w_in+ 2*padding[0]- dilation[0]*(kernel_size[0]- 1)- 1)/stride[0]+ 1)
            fc7_feat_w = np.floor((fc7_feat_w + 2 * 1 - 1 * (2 - 1) - 1) / 2 + 1)
            fc7_feat_h = np.floor((fc7_feat_h + 2 * 1 - 1 * (2 - 1) - 1) / 2 + 1)
        return fc7_feat_w, fc7_feat_h
        
##########################################################################################
#                                                                                        #
#                      dencoder definition using PyTorch                                 #
#                                                                                        #
##########################################################################################
class decoder(nn.Module):
    def __init__(self, n_deconvfilter, h_shape):
        print("initializing \"decoder\"")
        super(decoder, self).__init__()
        #3d conv7
        conv7_kernel_size = 3
        self.conv7 = Conv3d(in_channels= n_deconvfilter[0], \
                            out_channels= n_deconvfilter[1], \
                            kernel_size= conv7_kernel_size, \
                            padding = int((conv7_kernel_size - 1) / 2))
        
        
        #3d conv7
        conv8_kernel_size = 3
        self.conv8 = Conv3d(in_channels= n_deconvfilter[1], \
                            out_channels= n_deconvfilter[2], \
                            kernel_size= conv8_kernel_size, \
                            padding = int((conv8_kernel_size - 1) / 2))
        
        
        #3d conv7
        conv9_kernel_size = 3
        self.conv9 = Conv3d(in_channels= n_deconvfilter[2], \
                            out_channels= n_deconvfilter[3], \
                            kernel_size= conv9_kernel_size, \
                            padding = int((conv9_kernel_size - 1) / 2))
        
        
        #3d conv7
        conv10_kernel_size = 3
        self.conv10 = Conv3d(in_channels= n_deconvfilter[3], \
                            out_channels= n_deconvfilter[4], \
                            kernel_size= conv10_kernel_size, \
                            padding = int((conv10_kernel_size - 1) / 2))
        
        
        #3d conv7
        conv11_kernel_size = 3
        self.conv11 = Conv3d(in_channels= n_deconvfilter[4], \
                            out_channels= n_deconvfilter[5], \
                            kernel_size= conv11_kernel_size, \
                            padding = int((conv11_kernel_size - 1) / 2))
        
        #pooling layer
        self.unpool3d = Unpool3DLayer(unpool_size = 2)
        
        
        #nonlinearities of the network
        self.leaky_relu = LeakyReLU(negative_slope= 0.01)
        
    def forward(self, gru_out):
        gru_out_to_conv11 = nn.Sequential(self.unpool3d, self.conv7, self.leaky_relu, \
                                          self.unpool3d, self.conv8, self.leaky_relu, \
                                          self.unpool3d, self.conv9, self.leaky_relu, \
                                          self.conv10, self.leaky_relu, self.conv11)
        return gru_out_to_conv11(gru_out)

