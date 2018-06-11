#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 15:29:51 2018

@author: wangchu
"""
from __future__ import print_function
from past.builtins import xrange

from PIL import Image
import numpy as np
from random import randrange
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from models.res_gru_net import ResidualGRUNet
from lib.config import cfg
from lib.data_io import get_rendering_file, get_voxel_file
from lib.data_augmentation import preprocess_img

#functions to find image id
from z_img_id import max_IoU_category_id, max_mAP_category_id, \
                        min_loss_category_id,load_img, load_vox_model


###############################################################################
#                                                                             #
#                             gradient check                                  #
#                                                                             #
###############################################################################
def grad_check_sparse(f, net_x, net_y, param, analytic_grad, num_checks=10, h=1e-5):
  """
  sample a few random elements and only return numerical
  in this dimensions.
  """
  param_shape = param.shape.eval()
  #f_x = f(net_x, net_y)
  for i in xrange(num_checks):
    ix = tuple([randrange(m) for m in param_shape])
    param_val = param.get_value()
    oldval = param_val[ix]
    
    param_val[ix] = oldval + h # increment by h
    param.set_value(param_val)
    fxph = f(net_x, net_y) # evaluate f(x + h)
    
    param_val[ix] = oldval - h # increment by h
    param.set_value(param_val)
    fxmh = f(net_x, net_y) # evaluate f(x - h)
    
    param_val[ix] = oldval # reset
    param.set_value(param_val)
    
    grad_numerical = (fxph - fxmh) / (2 * h)
    grad_analytic = analytic_grad[ix]
    
    rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
    print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))

def compile_analytic_grad(net):
    print("\ncompiling analytic gradient function...")
    #compute the gradient of net.loss w.r.t first weight 
    grad_wrt_param = T.grad(net.loss, net.params[-5].val)
    return theano.function([net.x, net.y], grad_wrt_param)

def compile_loss_fn(net):
    print("\ncompiling loss function")
    return theano.function([net.x, net.y], net.loss)

if __name__ == '__main__':
    theano.config.floatX = 'float32'
    cfg.CONST.BATCH_SIZE = 1
    net = ResidualGRUNet(compute_grad=False)
    x = np.zeros((1,1,3,127,127), dtype=theano.config.floatX)
    y = np.zeros((1,32,2,32,32), dtype=theano.config.floatX)
    loss_fn = compile_loss_fn(net)
    loss = loss_fn(x, y)
    """feed an input image to the network"""
    """
    #get max mAP model id for each category
    result_file = "./output/ResidualGRUNet/default_model/test/result.mat"
    category_id_pair = min_loss_category_id(result_file)
    
    #get image id for a specific class
    category, model_id = category_id_pair[0]
    im = load_img(category, model_id, 0)
    voxel = load_vox_model(category, model_id)
    
    batch_voxel = np.zeros((1, voxel.shape[0], 2, voxel.shape[1], voxel.shape[2]),\
                           dtype=theano.config.floatX)
    
    batch_voxel[0, :, 0, :, :] = voxel < 1
    batch_voxel[0, :, 1, :, :] = voxel
    
    plt.imshow(im)
    plt.show()
    
    #reshape image to be compatible with network's input
    batch_im = im[np.newaxis, np.newaxis, :, :, :].transpose((0,1,4,2,3)).astype(theano.config.floatX)
    
    #create a GRUNet instance and compute the saliency map
    weights = "./output/ResidualGRUNet/default_model/weights.60000.npy"
    net = ResidualGRUNet(compute_grad=False)
    net.load(weights)
    
    #analytic grad function
    analytic_grad_fn = compile_analytic_grad(net)
    grad_wrt_w0 = analytic_grad_fn(batch_im, batch_voxel)
    
    loss_fn = compile_loss_fn(net)
    grad_check_sparse(loss_fn, batch_im, batch_voxel, net.params[-5].val, grad_wrt_w0)
    """











