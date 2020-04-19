import numpy as np

from models.base_gru_net import BaseGRUNet
from lib.config import cfg
from lib.layers import FCConv3DLayer_torch, Unpool3DLayer, \
                       SoftmaxWithLoss3D, BN_FCConv3DLayer_torch

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Linear, Conv2d, MaxPool2d, \
                     LeakyReLU, Conv3d, Tanh, Sigmoid, \
                     BatchNorm2d, BatchNorm3d

##########################################################################################
#                                                                                        #
#                      GRUNet definition using PyTorch                                   #
#                                                                                        #
##########################################################################################
class ResidualGRUNet(BaseGRUNet):
    def __init__(self):
        print("initializing \"ResidualGRUNet\"")
        super(ResidualGRUNet, self).__init__()
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
        self.conv1a = Conv2d(input_shape[1], n_convfilter[0], 7, padding=3)
        self.bn1a = BatchNorm2d(n_convfilter[0])
        self.conv1b = Conv2d(n_convfilter[0], n_convfilter[0], 3, padding=1)
        self.bn1b = BatchNorm2d(n_convfilter[0])
        
        #conv2
        self.conv2a = Conv2d(n_convfilter[0], n_convfilter[1], 3, padding=1)
        self.bn2a = BatchNorm2d(n_convfilter[1])
        self.conv2b = Conv2d(n_convfilter[1], n_convfilter[1], 3, padding=1)
        self.bn2b = BatchNorm2d(n_convfilter[1])
        self.conv2c = Conv2d(n_convfilter[0], n_convfilter[1], 1)
        self.bn2c = BatchNorm2d(n_convfilter[1])
        
        #conv3
        self.conv3a = Conv2d(n_convfilter[1], n_convfilter[2], 3, padding=1)
        self.bn3a = BatchNorm2d(n_convfilter[2])
        self.conv3b = Conv2d(n_convfilter[2], n_convfilter[2], 3, padding=1)
        self.bn3b = BatchNorm2d(n_convfilter[2])
        self.conv3c = Conv2d(n_convfilter[1], n_convfilter[2], 1)
        self.bn3c = BatchNorm2d(n_convfilter[2])
        
        #conv4
        self.conv4a = Conv2d(n_convfilter[2], n_convfilter[3], 3, padding=1)
        self.bn4a = BatchNorm2d(n_convfilter[3])
        self.conv4b = Conv2d(n_convfilter[3], n_convfilter[3], 3, padding=1)
        self.bn4b = BatchNorm2d(n_convfilter[3])
        
        #conv5
        self.conv5a = Conv2d(n_convfilter[3], n_convfilter[4], 3, padding=1)
        self.bn5a = BatchNorm2d(n_convfilter[4])
        self.conv5b = Conv2d(n_convfilter[4], n_convfilter[4], 3, padding=1)
        self.bn5b = BatchNorm2d(n_convfilter[4])
        
        #conv6
        self.conv6a = Conv2d(n_convfilter[4], n_convfilter[5], 3, padding=1)
        self.bn6a = BatchNorm2d(n_convfilter[5])
        self.conv6b = Conv2d(n_convfilter[5], n_convfilter[5], 3, padding=1)
        self.bn6b = BatchNorm2d(n_convfilter[5])
        
        
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
        #conv3d_filter_shape = (self.n_deconvfilter[0], self.n_deconvfilter[0], 3, 3, 3)
        self.t_x_s_update = BN_FCConv3DLayer_torch(n_fc_filters[0], conv3d_filter_shape, h_shape)
        self.t_x_s_reset = BN_FCConv3DLayer_torch(n_fc_filters[0], conv3d_filter_shape, h_shape)
        self.t_x_rs = BN_FCConv3DLayer_torch(n_fc_filters[0], conv3d_filter_shape, h_shape)
        
    def forward(self, x, h, u, time):
        #for recurrent batch normalization, time is the current time step
        """
        x is the input and the size of x is (batch_size, channels, heights, widths).
        h and u is the hidden state and activation of last time step respectively.
        This function defines the forward pass of the encoder of the network.
        """
        #modified conv1
        conv1a = self.conv1a(x)
        conv1a = self.bn1a(conv1a)
        rect1a = self.leaky_relu(conv1a)
        
        conv1b = self.conv1b(rect1a)
        conv1b = self.bn1b(conv1b)
        rect1 = self.leaky_relu(conv1b)
        pool1 = self.pool(rect1)
        
        
        #conv2
        conv2a = self.conv2a(pool1)
        conv2a = self.bn2a(conv2a)
        rect2a = self.leaky_relu(conv2a)
        conv2b = self.conv2b(rect2a)
        conv2b = self.bn2b(conv2b)
        #residual connection between pool1 and conv2b
        conv2c = self.conv2c(pool1)
        conv2c = self.bn2c(conv2c)
        res2 = conv2c + conv2b
        rect2 = self.leaky_relu(res2)
        pool2 = self.pool(rect2)
        
        
        #conv3
        conv3a = self.conv3a(pool2)
        conv3a = self.bn3a(conv3a)
        rect3a = self.leaky_relu(conv3a)
        conv3b = self.conv3b(rect3a)
        conv3b = self.bn3b(conv3b)
        #residual connection between pool1 and conv2b
        conv3c = self.conv3c(pool2)
        conv3c = self.bn3c(conv3c)
        res3 = conv3c + conv3b
        rect3 = self.leaky_relu(res3)
        pool3 = self.pool(rect3)
        
        
        #conv4
        conv4a = self.conv4a(pool3)
        conv4a = self.bn4a(conv4a)
        rect4a = self.leaky_relu(conv4a)
        conv4b = self.conv4b(rect4a)
        conv4b = self.bn4b(conv4b)
        res4 = pool3 + conv4b
        rect4 = self.leaky_relu(res4)
        pool4 = self.pool(rect4)
        
        
        #conv5
        conv5a = self.conv5a(pool4)
        conv5a = self.bn5a(conv5a)
        rect5a = self.leaky_relu(conv5a)
        conv5b = self.conv5b(rect5a)
        conv5b = self.bn5b(conv5b)
        res5 = pool4 + conv5b
        rect5 = self.leaky_relu(res5)
        pool5 = self.pool(rect5)
        
        
        #conv6
        conv6a = self.conv6a(pool5)
        conv6a = self.bn6a(conv6a)
        rect6a = self.leaky_relu(conv6a)
        conv6b = self.conv6b(rect6a)
        conv6b = self.bn6b(conv6b)
        #residual connection between pool1 and conv2b
        res6 = pool5 + conv6b
        rect6 = self.leaky_relu(res6)
        pool6 = self.pool(rect6)
        
        
        pool6 = pool6.view(pool6.size(0), -1)
        
        
        fc7 = self.fc7(pool6)
        rect7 = self.leaky_relu(fc7)
        
        t_x_s_update = self.t_x_s_update(rect7, h, time)
        t_x_s_reset = self.t_x_s_reset(rect7, h, time)
        
        update_gate = self.sigmoid(t_x_s_update)
        complement_update_gate = 1 - update_gate
        reset_gate = self.sigmoid(t_x_s_reset)
        
        rs = reset_gate * h
        t_x_rs = self.t_x_rs(rect7, rs, time)
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
        self.conv7a = Conv3d(n_deconvfilter[0], n_deconvfilter[1], 3, padding=1)
        self.bn7a = BatchNorm3d(n_deconvfilter[1])
        self.conv7b = Conv3d(n_deconvfilter[1], n_deconvfilter[1], 3, padding=1)
        self.bn7b = BatchNorm3d(n_deconvfilter[1])
        
        #3d conv8
        self.conv8a = Conv3d(n_deconvfilter[1], n_deconvfilter[2], 3, padding=1)
        self.bn8a = BatchNorm3d(n_deconvfilter[2])
        self.conv8b = Conv3d(n_deconvfilter[2], n_deconvfilter[2], 3, padding=1)
        self.bn8b = BatchNorm3d(n_deconvfilter[2])
        
        
        #3d conv9
        self.conv9a = Conv3d(n_deconvfilter[2], n_deconvfilter[3], 3, padding=1)
        self.bn9a = BatchNorm3d(n_deconvfilter[3])
        self.conv9b = Conv3d(n_deconvfilter[3], n_deconvfilter[3], 3, padding=1)
        self.bn9b = BatchNorm3d(n_deconvfilter[3])
        self.conv9c = Conv3d(n_deconvfilter[2], n_deconvfilter[3], 1)
        self.bn9c = BatchNorm3d(n_deconvfilter[3])
        
        #3d conv10
        self.conv10a = Conv3d(n_deconvfilter[3], n_deconvfilter[4], 3, padding=1)
        self.bn10a = BatchNorm3d(n_deconvfilter[4])
        self.conv10b = Conv3d(n_deconvfilter[4], n_deconvfilter[4], 3, padding=1)
        self.bn10b = BatchNorm3d(n_deconvfilter[4])
        self.conv10c = Conv3d(n_deconvfilter[4], n_deconvfilter[4], 3, padding=1)
        self.bn10c = BatchNorm3d(n_deconvfilter[4])
        self.conv10d = Conv3d(n_deconvfilter[3], n_deconvfilter[4], 1)
        self.bn10d = BatchNorm3d(n_deconvfilter[4])
        
        #3d conv11
        self.conv11 = Conv3d(n_deconvfilter[4], n_deconvfilter[5], 3, padding=1)
        
        #unpooling layer
        self.unpool3d = Unpool3DLayer(unpool_size = 2)
        
        
        #nonlinearities of the network
        self.leaky_relu = LeakyReLU(negative_slope= 0.01)
        
    def forward(self, gru_out):
        unpool7 = self.unpool3d(gru_out)
        conv7a = self.conv7a(unpool7)
        conv7a = self.bn7a(conv7a)
        rect7a = self.leaky_relu(conv7a)
        conv7b = self.conv7b(rect7a)
        conv7b = self.bn7b(conv7b)
        #residual connection before nonlinearity
        res7 = unpool7 + conv7b
        rect7 = self.leaky_relu(res7)
        
        
        unpool8 = self.unpool3d(rect7)
        conv8a = self.conv8a(unpool8)
        conv8a = self.bn8a(conv8a)
        rect8a = self.leaky_relu(conv8a)
        conv8b = self.conv8b(rect8a)
        conv8b = self.bn8b(conv8b)
        #residual connection before nonlinearity
        res8 = unpool8 + conv8b
        rect8 = self.leaky_relu(res8)
        
        
        unpool9 = self.unpool3d(rect8)
        conv9a = self.conv9a(unpool9)
        conv9a = self.bn9a(conv9a)
        rect9a = self.leaky_relu(conv9a)
        conv9b = self.conv9b(rect9a)
        conv9b = self.bn9b(conv9b)
        conv9c = self.conv9c(unpool9)
        conv9c = self.bn9c(conv9c)
        #residual connection before nonlinearity
        res9 = conv9c + conv9b
        rect9 = self.leaky_relu(res9)
        
        
        conv10a = self.conv10a(rect9)
        conv10a = self.bn10a(conv10a)
        rect10a = self.leaky_relu(conv10a)
        conv10b = self.conv10b(rect10a)
        conv10b = self.bn10b(conv10b)
        rect10b = self.leaky_relu(conv10b)
        conv10c = self.conv10c(rect10b)
        #residual connection before nonlinearity
        conv10d = self.conv10d(rect9)
        conv10d = self.bn10d(conv10d)
        res10 = conv10c + conv10d
        rect10 = self.leaky_relu(res10)
        
        conv11 = self.conv11(rect10)
        return conv11
