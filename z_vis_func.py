#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 15:57:33 2018

@author: wangchu
"""
import os
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from PIL import Image

from models.gru_net import GRUNet
from models.res_gru_net import ResidualGRUNet
from lib.config import cfg
from lib.data_io import get_rendering_file
from lib.data_augmentation import preprocess_img

#functions to find image id
from z_img_id import max_IoU_category_id, max_mAP_category_id, \
                        min_loss_category_id, load_img

def loss_moving_average(loss):
    #1460 is the number of iterations that all the training examples are seen once
    kernal = np.ones((1460,))
    return np.convolve(loss, kernal, mode='valid')/1460

###############################################################################
#                                                                             #
#            visualize learnable parameters of the network                    #
#                                                                             #
###############################################################################
from past.builtins import xrange
from math import sqrt, ceil

def visualize_grid(Xs, ubound=255.0, padding=1):
  """
  Reshape a 4D tensor of image data to a grid for easy visualization.

  Inputs:
  - Xs: Data of shape (N, H, W, C)
  - ubound: Output grid will have values scaled to the range [0, ubound]
  - padding: The number of blank pixels between elements of the grid
  """
  (N, H, W, C) = Xs.shape
  grid_size = int(ceil(sqrt(N)))
  grid_height = H * grid_size + padding * (grid_size - 1)
  grid_width = W * grid_size + padding * (grid_size - 1)
  grid = np.zeros((grid_height, grid_width, C))
  next_idx = 0
  y0, y1 = 0, H
  for y in xrange(grid_size):
    x0, x1 = 0, W
    for x in xrange(grid_size):
      if next_idx < N:
        img = Xs[next_idx]
        low, high = np.min(img), np.max(img)
        grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
        # grid[y0:y1, x0:x1] = Xs[next_idx]
        next_idx += 1
      x0 += W + padding
      x1 += W + padding
    y0 += H + padding
    y1 += H + padding
  # grid_max = np.max(grid)
  # grid_min = np.min(grid)
  # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
  return grid


def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()
    
def test_show():
    weights = "/Users/wangchu/Desktop/full_gru_output/default_model/weights.60000.npy"
    params = np.load(weights)
    first_layer_w = params[0].transpose(0, 2, 3, 1)
    plt.figure(figsize=(20, 20))
    plt.imshow(visualize_grid(first_layer_w, padding=1).astype('uint8'))
    plt.gca().axis('off')
    plt.show()
    #plt.savefig("/Users/wangchu/Desktop/filters.pdf")
    #plot_param(first_layer_w, show=True)
    
###############################################################################
#                                                                             #
#                           occlusion experiments                             #
#                                                                             #
###############################################################################
from z_img_id import load_img, load_vox_model
from z_net_eval import compile_loss_fn
#generate occluded images, the shape of input image is (height, width, channel)
def gen_occ_img(img, occ_h=10, occ_w=10):
    img_h, img_w, img_c = img.shape
    #create an occlude window
    occ = np.zeros((occ_h, occ_w, img_c))
    occ.fill(240)
    
    new_h = img_h - occ_h
    new_w = img_w - occ_w
    occ_img_array = np.zeros((new_h, new_w, img_h, img_w, img_c), dtype=theano.config.floatX)
    for i in range(occ_img_array.shape[0]):
        for j in range(occ_img_array.shape[1]):
            img_copy = img.copy()
            img_copy[i:i+occ_h, j:j+occ_w] = occ
            occ_img_array[i, j, :, :, :] = img_copy
    return occ_img_array, new_h, new_w

def gen_heat_map(loss_fn, img, voxel, show):
    batch_size = cfg.CONST.BATCH_SIZE
    occ_img_array, new_h, new_w = \
                        gen_occ_img(img, occ_h=3, occ_w=3)
    #make image compatible with network
    occ_img_array = occ_img_array.transpose(0, 1, 4, 2, 3)
    occ_img_array = occ_img_array[:, np.newaxis, :, :,:,:]
    
    heat_array = np.zeros((new_h, new_w))
    #iterate over columns for each row
    for i in range(new_h):
        for j in range(new_w // batch_size):
            x = occ_img_array[i, :, j*batch_size:(j+1)*batch_size]
            #print("i is %s, j is %s, computing loss" % (i, j))
            heat_array[i, j*batch_size:(j+1)*batch_size] = loss_fn(x, voxel)
    if show:
        plt.imshow(heat_array, cmap='hot', interpolation='nearest')
        plt.savefig("heat_map.pdf")
    return heat_array

def compute_heat_map():
    
    cfg.CONST.BATCH_SIZE = 31
    theano.config.floatX = 'float32'
    
    result_file = "./output/ResidualGRUNet/default_model/test/result.mat"
               
    weights = "./output/ResidualGRUNet/default_model/weights.60000.npy"
    net = ResidualGRUNet(compute_grad=False)
    net.load(weights)
    
    """only need to compile the loss function once!!!"""
    loss_fn = compile_loss_fn(net)
    
    name_dict = {0: "max_mAP", 1: "min_loss", 2: "max_IoU"}
    for i in range(3):
        if i == 0:
            category_model_id_pair = max_mAP_category_id(result_file)
        if i == 1:
            category_model_id_pair = min_loss_category_id(result_file)
        if i == 2:
            category_model_id_pair = max_IoU_category_id(result_file)[str(cfg.TEST.VOXEL_THRESH[0])]
        for j in range(13):
            category, model_id = category_model_id_pair[j]
            img = load_img(category, model_id, 1, train=False)
            
            voxel = load_vox_model(category, model_id)
            
            batch_voxel = np.zeros((1, voxel.shape[0], 2, voxel.shape[1], voxel.shape[2]),\
                                   dtype=theano.config.floatX)
            
            batch_voxel[0, :, 0, :, :] = voxel < 1
            batch_voxel[0, :, 1, :, :] = voxel
            
            vox = np.repeat(batch_voxel,cfg.CONST.BATCH_SIZE, axis=0)
            
            
            heat_array = gen_heat_map(net, img, vox, show=False)
            np.save("%s_heat_array_category_%s" % (name_dict[i], category), heat_array)


###############################################################################
#                                                                             #
#                 plot heat map and corresponding image                       #
#                                                                             #
###############################################################################

#get the index of the second "underline" notation
def get2nd_(map_name):
    underline1 = map_name.index("_") + 1
    underline2 = underline1 + map_name[underline1 :].index("_")
    return underline2
#return a list of pairs of (img, heat_map)
def img_heat_map_pair(map_names, map_dir, result_file):
    #a dictionary used to load image
    img_type_cat_dict = {"max_mAP": dict((category, img_id) \
                                     for category, img_id in max_mAP_category_id(result_file)),\
                         "min_loss": dict((category, img_id) \
                                     for category, img_id in min_loss_category_id(result_file)),\
                         "max_IoU": dict((category, img_id) \
                                     for category, img_id in max_IoU_category_id(result_file)[str(cfg.TEST.VOXEL_THRESH[0])])}
    img_heat_map_pair = []
    #find the image according to the image type and model id
    for map_name in map_names:
        img_type = map_name[: get2nd_(map_name)]
        model_id = map_name[-12 : -4]
        img_id = img_type_cat_dict[img_type][model_id]
        img = load_img(model_id, img_id, 0, train=False)
        heat_map = np.load(os.path.join(map_dir, map_name))
        img_heat_map_pair.append((img, heat_map))
    return img_heat_map_pair


def get_img_heat_map(map_dir, result_file):
    if os.path.isdir(map_dir):
        map_names = [name for name in os.listdir(map_dir) if name[-4:] == ".npy"]
        return img_heat_map_pair(map_names, map_dir, result_file)
    else:
        raise Exception("map_dir must be a directory ")
    
    
def draw_img_heat_map(map_dir, result_file):
    #get (image, heat_map) pair
    img_heat_map_pair = get_img_heat_map(map_dir, result_file)
    
    num_pair = len(img_heat_map_pair)
    
    
    interval = int(np.ceil(np.sqrt(num_pair)))
    
    row = interval
    col = interval * 2
    
    plt.figure(figsize=(20, 20))
    plt.title("image and the corresponding heat map")
    
    print("\nploting the image and heat map pair")
    for i in range(row):
            for j in range(1, interval + 1):
                try:
                    image, heat_map = img_heat_map_pair[i + j - 1]
                    image[:20, :20] = [240, 240, 240]
                    
                    plt.subplot(row, col, i * col + 2 * j - 1)
                    plt.imshow(image)
                    plt.gca().axis('off')
                    
                    plt.subplot(row, col, i * col + 2 * j)
                    plt.imshow(heat_map)
                    plt.gca().axis('off')
                    
                except IndexError:
                    pass
    plt.tight_layout()
    plt.show()

def plot_img_heat_map_pair():
    
    result_file = "./output/ResidualGRUNet/default_model/test/result.mat"
    map_dir = "./z_visual_data/heat_map_array"
    draw_img_heat_map(map_dir, result_file)
    


###############################################################################
#                                                                             #
#                 compile and compute saliency map                            #
#                                                                             #
###############################################################################
def compile_saliency_map(net):
    """
    compile a function to compute the saliency map and predicted classes 
    for a given minibatch of input images.
    """
    print("\ncompiling saliency map function...")
    #the shape of self.x is (n_views, batch_size, color_channels, img_h, img_w)
    inp = net.x
    #the shape of conv11_output is (batch_size, 32, 2, 32, 32)
    outp_neg = net.conv11_output[:, :, 0, :, :].sum()
    outp_pos = net.conv11_output[:, :, 1, :, :].sum()
    #find which part affect the "not occupied" scores
    saliency_neg = theano.grad(outp_neg, wrt = inp)
    #find which part affect the "occupied" scores
    saliency_pos = theano.grad(outp_pos, wrt = inp)
    #not a classification problem, no need to compute max class
    return theano.function([inp], [saliency_neg, saliency_pos])
 
def compute_saliency_map():
    
    theano.config.floatX = 'float32'
    cfg.CONST.BATCH_SIZE = 1
    
    #create a GRUNet instance and compute the saliency map
    weights = "./output/ResidualGRUNet/default_model/weights.60000.npy"
    net = ResidualGRUNet()
    net.load(weights)
    
    saliency_fn = compile_saliency_map(net)
    
def plot_saliency_map(saliency_fn):
    #get max mAP model id for each category
    result_file = "./output/ResidualGRUNet/default_model/test/result.mat"
   
    max_mAP_category_model_id_pair = max_mAP_category_id(result_file)
    
    num_pair = len(max_mAP_category_model_id_pair)
    
    interval = int(np.ceil(np.sqrt(num_pair)))
    
    row = interval
    col = interval * 2
    
    plt.figure(figsize=(20, 20))
    plt.title("image and the corresponding saliency map")
    
    for i, (category, model_id) in enumerate(max_mAP_category_model_id_pair):
        img = load_img(category, model_id, 0, train=False)
        #reshape image to be compatible with network's input
        img_input = img[np.newaxis, np.newaxis, :, :, :]
        img_input = img_input.transpose((0,1,4,2,3)).astype(theano.config.floatX)
        
        saliency_pos = saliency_fn(img_input)[1]
        saliency_pos = saliency_pos[0,0,:,:,:].transpose((1, 2, 0))
    
        plt.subplot(row, col, 2 * (i + 1) -1)
        plt.imshow(img)
        #plt.gca.axis('off')
        
        plt.subplot(row, col, 2 * (i + 1))
        plt.imshow(np.abs(saliency_pos).max(axis=-1), cmap='gray')
        #plt.gca.axis('off')
    
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    
    
    theano.config.floatX = 'float32'
    cfg.CONST.BATCH_SIZE = 1
    
    #create a GRUNet instance and compute the saliency map
    weights = "./output/ResidualGRUNet/default_model/weights.60000.npy"
    net = ResidualGRUNet()
    net.load(weights)
    
    saliency_fn = compile_saliency_map(net)
    
    
    
    
    