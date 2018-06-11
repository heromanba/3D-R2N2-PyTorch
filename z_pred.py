#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:11:25 2018

@author: wangchu
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lib.config import cfg
from lib.data_augmentation import preprocess_img
from lib.voxel import voxel2obj
from lib.solver import Solver
from models.res_gru_net import ResidualGRUNet
#the directory name of images are the time they are taken
def load_imgs(taken_time):
    assert(os.path.isdir(taken_time))
    
    img_h = cfg.CONST.IMG_H
    img_w = cfg.CONST.IMG_W
    
    imgs = []
    
    for file in os.listdir(taken_time):
        if file[-4:] in [".JPG",".png"]:
            img = Image.open(os.path.join(taken_time, file))
            img = img.resize((img_h, img_w), Image.ANTIALIAS)
            img = preprocess_img(img, train=False)
            imgs.append([np.array(img).transpose( \
                         (2, 0, 1)).astype(np.float32)])
            plt.imshow(img)
            plt.show()
    return np.array(imgs)
            

def pred_show_3D_model(imgs_dir, solver):
    assert(os.path.isdir(imgs_dir))
    
    for taken_time in os.listdir(imgs_dir):
        taken_time = os.path.join(imgs_dir, taken_time)
        if os.path.isdir(taken_time):
            imgs = load_imgs(taken_time)
            vox_pred, _ = solver.test_output(imgs)
            #save the voxel prediction to an OBJ file
            obj_fn = os.path.join(taken_time, "prediction.obj")
            vox_pred_thresholded = vox_pred[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH
            #save obj file
            voxel2obj(obj_fn, vox_pred_thresholded)
            

if __name__ == '__main__':
    import theano
    theano.config.floatX = 'float32'
    
    
    DEFAULT_WEIGHTS = './output/ResidualGRUNet/default_model/weights.npy'


    net = ResidualGRUNet(compute_grad = False)
    net.load(DEFAULT_WEIGHTS)
    
    solver = Solver(net)
    
    
    real_imgs_dir = "./z_visual_data/real_world_imgs"
    
    pred_show_3D_model(real_imgs_dir, solver)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    