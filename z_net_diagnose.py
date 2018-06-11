
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:52:11 2018

@author: wangchu
"""
import theano
import numpy as np
import matplotlib.pyplot as plt



###############################################################################
#                                                                             #
#                 check activations per nonlinearity layer                    #
#                                                                             #
###############################################################################
activs_dict = {0: "rect1", 1: "rect2", 2: "rect3", 3: "rect4", \
               4: "rect5", 5: "rect6", 6: "rect7_", 7: "update_gate_output",\
               8:"reset_gate", 9: "tanh_t_x_rs", 10: "rect7", \
               11: "rect8", 12: "rect9", 13: "rect10"}


def compile_nonlin_activs_fn(net):
    print("\ncompiling the nonlin_activs_fn")
    fn = theano.function([net.x], \
                         [net.rect1_output, net.rect2_output, \
                          net.rect3_output, net.rect4_output, \
                          net.rect5_output, net.rect6_output,\
                          net.rect7_output1, net.update_gate_output,\
                          net.reset_gate_, net.tanh_t_x_rs_output,\
                          net.rect7_output2, net.rect8_output,\
                          net.rect9_output, net.rect10_output])
    return fn

def nonlin_activs_hist(nonlin_activs):
    print("\nploting histograms")
    
    num_act = len(nonlin_activs) + 2    #two more subplots for mean and std
    row = int(np.ceil(np.sqrt(num_act)))
    col = int(np.ceil(num_act / row))
    
    plt.figure(figsize=(20,20))
    plt.title("Activation distribution of every nonlinearity")
    
    means = []
    stds = []
    
    for i in range(row):
            for j in range(1, col + 1):
                try:
                    activ = nonlin_activs[i * col + j - 1]
                    mean = np.mean(activ)
                    std = np.std(activ)
                    
                    means.append(mean)
                    stds.append(std)
                    
                    plt.subplot(row, col, i * col + j)
                    plt.title("layer %s\nmean: %s\nstd:%s" % \
                             (activs_dict[i * col + j - 1], mean, std))
                    #print(activ)
                    if np.isnan(np.max(np.abs(activ))):
                        print("layer %s\nmean: %s\nstd:%s" % \
                             (activs_dict[i * col + j - 1], mean, std))
                        break
                    plt.hist(activ.ravel(), bins = 100, density = True)
                    
                except IndexError:
                    break
    
    #draw plots of mean and std at last
    plt.subplot(row, col, row * col -1)
    plt.title("means of each layer")
    plt.plot(np.array(means))
    
    plt.subplot(row, col, row * col)
    plt.title("standard deviations of each layer")
    plt.plot(np.array(stds))
    
    plt.tight_layout()
    plt.show()

def draw_activs_plots():
    
    theano.config.floatX = 'float32'
    
    x = np.random.randn(5, 36, 3, 127, 127).astype(theano.config.floatX)
    y = np.random.randn(36, 32, 2, 32, 32).astype(theano.config.floatX)
    
    net = GRUNet()
    
    nonlin_activs_fn = compile_nonlin_activs_fn(net)
    
    nonlin_activs = nonlin_activs_fn(x)

    nonlin_activs_hist(nonlin_activs)
    

###############################################################################
#                                                                             #
#                 draw curve of ratio of weights: updates                     #
#                                                                             #
###############################################################################
import os
#get iteration index based on the format of "weights.index.npy"
def get_iter_id(weight_name):
    dot1 = weight_name.index(".") + 1
    dot2 = dot1 + weight_name[dot1 + 1:].index(".") + 1
    return dot1, dot2

#transform names list to dictionary to be sorted
def list_2_dict_ids(names_list):
    iter_ids = []
    names_dict = {}
    for name in names_list:
        dot1, dot2 = get_iter_id(name)
        iter_id = int(name[dot1: dot2])
        #key is iteration index, value is file name
        names_dict[iter_id] = name
        iter_ids.append(iter_id)
    iter_ids.sort()
    return iter_ids, names_dict

#return a list of weight file names in incremental order
def get_w_ids_path(weights_dir):
    if os.path.isdir(weights_dir):
        #get all npy files
        names_list = [name for name in os.listdir(weights_dir) if name[-4:] == ".npy"]
        #get the sorted index
        iter_ids, names_dict = list_2_dict_ids(names_list)
        #add weights directory to weight file names to get weight paths
        for iter_id in iter_ids: 
            names_dict[iter_id] = os.path.join(weights_dir, names_dict[iter_id])
        return iter_ids, names_dict
    else:
        raise Exception("file_path must be a directory")
        
    
#read all the saved weights in the given path and plot ratio curves
def w_update_ids_ratio(weights_dir):
    
    iter_ids, names_dict = get_w_ids_path(weights_dir)
    
    #a list to store the paramters from all iterations
    all_iter_params = []
    
    print("\nreading weights files")
    
    for iter_id in iter_ids:
        params = np.load(names_dict[iter_id])
        all_iter_params.append(params)
    
    #switch the parameters to be the first index
    #note: params_array.shape=(number of parameters, number of iterations)   
    params_array = np.array(all_iter_params).transpose()
    
    update_array = params_array[:, 1:] - params_array[:, :-1]
    
    params_array = params_array[:, :-1]
    
    num_param = update_array.shape[0]
    num_iter = update_array.shape[1]
    
    ratio = []
    
    print("\nprocessing the ratio data")
    #process the ratio data
    for i in range(num_param):
        ratio_i = []
        for j in range(num_iter):
            #get the ratio for i_th parameter at j_th iteration
            update_scale = np.linalg.norm(update_array[i, j].ravel())
            param_scale = np.linalg.norm(params_array[i, j].ravel())
            ratio_ij = update_scale / param_scale
            ratio_i.append(ratio_ij)
        ratio.append(ratio_i)
    return iter_ids, np.array(ratio)
    
    

def plot_ratio_curve(weights_dir):
    iter_ids, ratio = w_update_ids_ratio(weights_dir)
    
    iter_ids_str = [str(iter_id) for iter_id in iter_ids]
    
    num_params = ratio.shape[0]
    
    plt.figure(figsize=(20, 20))
    plt.title("update ratio mean of all parameters")
    
    col = min(int(np.ceil(np.sqrt(num_params))), 5)
    row = int(np.ceil(num_params / col))
    
    print("\nploting the ratio")
    for i in range(row):
            for j in range(1, col + 1):
                try:
                    ratio_ij = ratio[i * col + j - 1]
                    
                    plt.subplot(row, col, i * col + j)
                    plt.title("update ratio mean %s th parameter" % (i * col + j - 1))
                    plt.ylabel("update ratio mean")
                    plt.xlabel("iteration")
                    plt.plot(ratio_ij)
                except IndexError:
                    pass
    plt.tight_layout()
    plt.show()
    
    
if __name__ == '__main__':
    """
    from lib.config import cfg
    theano.config.floatX = 'float32'
    cfg.CONST.BATCH_SIZE = 24
    #ratio = plot_ratio_curve("./z_visual_data/GRU_state_of_illness/GRU前4000weights")
    #draw_activs_plots()

    from lib.solver import Solver
    
    x = np.random.randn(5, 24, 3, 127, 127).astype(theano.config.floatX)
    y = np.random.randn(24, 32, 2, 32, 32).astype(theano.config.floatX)
    
    DEFAULT_WEIGHTS = '/Users/wangchu/Desktop/default_model/weights.30000.npy'

    net = ResidualGRUNet()
    
    solver = Solver(net)

    loss = solver.train_loss(x, y)
    """
    import torch
    from torch.autograd import Variable, gradcheck
    import torch.nn as nn
    from lib.solver import Solver
    from lib.layers import SoftmaxWithLoss3D
    from models.res_gru_net import ResidualGRUNet
    net = ResidualGRUNet()
    
    x = Variable(torch.FloatTensor(5, 36, 3, 127, 127).fill_(1))
    out = net(x)
    
    """
    a = Variable(torch.IntTensor(4, 4).zero_())
    
    b = torch.FloatTensor([[11, 12], [21, 22]])
    
    a[0:a.size(0):2, 0:a.size(1):2] = b
    """
    
    
    





