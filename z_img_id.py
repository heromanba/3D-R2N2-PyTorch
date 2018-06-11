from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from lib.data_io import category_model_id_dict, get_rendering_file, get_voxel_file
from lib.config import cfg
from lib.data_augmentation import preprocess_img
from lib.binvox_rw import read_as_3d_array

###############################################################################
#                                                                             #
#                       images with minimal loss                              #
#                                                                             #
###############################################################################
def min_loss_category_id(result_file, n_th=0):
    
    #load ".mat" file from "../default_model/test" directory
    test_output = sio.loadmat(result_file)
    
    #a list of dictionaries to find category_model_id of minimum loss object
    data_dict_list = category_model_id_dict(dataset_portion=cfg.TEST.DATASET_PORTION)
    
    min_loss_category_model_id_pair = []
    #iterate over 13 categories
    for cat_dict in data_dict_list:
        
        #find losses corresponding to a sepcific category
        range_in_test = cat_dict['range_in_test']
        
        #note that the shape of "test_output['cost']" is (1, 8762) when test_dataset_portion is [0.8, 1]
        cat_cost = test_output['cost'][0, range_in_test[0]:range_in_test[1]]
        
        #find the index of the minimum loss
        min_loss_index = np.argsort(cat_cost)[n_th]
        
        #find the model id of the minimum loss
        min_loss_id = cat_dict['portioned_model_ids'][min_loss_index]
        min_loss_category_model_id_pair.append((cat_dict['category_id'], min_loss_id))
    
    return min_loss_category_model_id_pair

###############################################################################
#                                                                             #
#                       images with maximal mAP                               #
#                                                                             #
###############################################################################
def max_mAP_category_id(result_file, n_th=-1):
    
    #load ".mat" file from "../default_model/test" directory
    test_output = sio.loadmat(result_file)
    
    #a list of dictionaries to find category_model_id of minimum loss object
    data_dict_list = category_model_id_dict(dataset_portion=cfg.TEST.DATASET_PORTION)
    
    max_mAP_category_model_id_pair = []
    #iterate over 13 categories
    for cat_dict in data_dict_list:
        
        #find mAPs corresponding to a sepcific category
        range_in_test = cat_dict['range_in_test']
        
        #note that the shape of "test_output['mAP']" is (8762, 1) when test_dataset_portion is [0.8, 1]
        cat_mAP = test_output['mAP'][range_in_test[0]:range_in_test[1], 0]
        
        #find the index of the maximum mAP
        max_mAP_index = np.argsort(cat_mAP)[n_th]
        
        #find the model id of the maximum mAP
        max_mAP_id = cat_dict['portioned_model_ids'][max_mAP_index]
        max_mAP_category_model_id_pair.append((cat_dict['category_id'], max_mAP_id))
    return max_mAP_category_model_id_pair

###############################################################################
#                                                                             #
#                       images with maximal IoU                               #
#                                                                             #
###############################################################################

def max_IoU_category_id(result_file, n_th=-1):
    #load ".mat" file from "../default_model/test" directory
    test_output = sio.loadmat(result_file)
    
    #a list of dictionaries to find category_model_id of minimum loss object
    data_dict_list = category_model_id_dict(dataset_portion=cfg.TEST.DATASET_PORTION)
    
    max_IoU_category_model_id_pair_per_thresh = {}
    
    #iterate over every threshold
    for thresh in cfg.TEST.VOXEL_THRESH:
        test_output_per_thresh = test_output[str(thresh)]
        
        max_IoU_category_model_id_pair = []
        #iterate over 13 categories
        for cat_dict in data_dict_list:
            
            #find evaluation corresponding to a sepcific category
            range_in_test = cat_dict['range_in_test']
            
            #note that the shape of "test_output_per_thresh" is (8762, 1, 5) when test_dataset_portion is [0.8, 1]
            cat_evaluate_voxel_prediction = \
                        test_output_per_thresh[range_in_test[0]:range_in_test[1], 0, :]
            
            #IoU = intersection / union
            IoU = cat_evaluate_voxel_prediction[:, 1] / cat_evaluate_voxel_prediction[:, 2]
            
            #find the index of the maximum IoU
            max_IoU_index = np.argsort(IoU)[n_th]
            
            #find the model id of the maximum IoU
            max_IoU_id = cat_dict['portioned_model_ids'][max_IoU_index]
            max_IoU_category_model_id_pair.append((cat_dict['category_id'], max_IoU_id))
        
        max_IoU_category_model_id_pair_per_thresh[str(thresh)] = max_IoU_category_model_id_pair
    return max_IoU_category_model_id_pair_per_thresh

###############################################################################
#                                                                             #
#                         load and process data                               #
#                                                                             #
###############################################################################
def load_img(category, model_id, img_id, train=False):
    img_fn = get_rendering_file(category, model_id, img_id)
    im = Image.open(img_fn)
    
    t_im = preprocess_img(im, train=train)
    return t_im
    
def load_vox_model(category, model_id):
    vox_fn = get_voxel_file(category, model_id)
    with open(vox_fn, 'rb') as f:
        voxel = read_as_3d_array(f)
    return voxel.data

def loss_moving_average(loss):
    #1460 is the number of iterations that all the training examples are seen once
    kernal = np.ones((1460,))
    return np.convolve(loss, kernal, mode='valid')/1460

if __name__ == '__main__':
    from lib.voxel import voxel2obj
    import theano
    #amsgrad_test_output = sio.loadmat("/Users/wangchu/Desktop/output_of_amsgrad/GRUNet/default_model/test/result.mat")
    #airplane_test_output = sio.loadmat("/Users/wangchu/Desktop/GRUNet_airplane_dataset/default_model/test/result.mat")
    #car_test_output = sio.loadmat("/Users/wangchu/Desktop/GRUNet_car_dataset/default_model/test/result.mat")
    #gru_test_output = sio.loadmat("/Users/wangchu/Desktop/GRUNet_full_dataset/default_model/test/result.mat")
    #gru_test_output = sio.loadmat("/Users/wangchu/Desktop/output/GRUNet/default_model/test/result.mat")
    """
    result_file = "./output/ResidualGRUNet/default_model/test/result.mat"
    
    max_mAP_category_model_id_pair = max_mAP_category_id(result_file)#[str(cfg.TEST.VOXEL_THRESH[0])]
    category, model_id = max_mAP_category_model_id_pair[4]
    im = load_img(category, model_id, 1, train=False)
    plt.imshow(im)
    plt.gca().axis('off')
    plt.savefig("/Users/wangchu/Desktop/watercraft_max_mAP.jpg")
    
    voxel = load_vox_model(category, model_id)
    
    batch_voxel = np.zeros((1, voxel.shape[0], 2, voxel.shape[1], voxel.shape[2]),\
                           dtype=theano.config.floatX)
    
    batch_voxel[0, :, 0, :, :] = voxel < 1
    batch_voxel[0, :, 1, :, :] = voxel
    
    voxel2obj("watercraft_max_mAP.obj",batch_voxel[0, :, 1, :, :])
    """
    
    """
    loss = np.loadtxt("/Users/wangchu/Desktop/output/GRUNet/default_model/loss.60000.txt")
    plt.plot(loss_moving_average(loss))
    plt.grid(True)
    plt.show()
    """
    
    loss_Res = np.loadtxt("/Users/wangchu/Desktop/ResGRU_output_from_GD/ResidualGRUNet/default_model/loss.60000.txt")
    loss = np.loadtxt("/Users/wangchu/Desktop/GRUNet_full_dataset/default_model/loss.60000.txt")
    
    plt.subplot(211)
    plt.plot(loss_moving_average(loss_Res))
    plt.ylabel('ResGRULoss')
    plt.xlabel('iteration')
    #axes = plt.gca()
    #axes.set_ylim([0, 0.3])
    plt.grid(True)
    
    plt.subplot(212)
    plt.plot(loss_moving_average(loss))
    plt.ylabel('GRUloss')
    plt.xlabel('iteration')
    #axes = plt.gca()
    #axes.set_ylim([0, 0.3])
    plt.grid(True)
    plt.savefig("/Users/wangchu/Desktop/loss_moving_average.jpg")
    
