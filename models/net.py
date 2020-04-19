import numpy as np
import datetime as dt

from lib.config import cfg
from lib.utils import weight_init

import torch.nn as nn




class Net(nn.Module):

    def __init__(self, random_seed=dt.datetime.now().microsecond, compute_grad=True):
        print("initializing \"Net\"")
        super(Net, self).__init__()
        self.rng = np.random.RandomState(random_seed)

        self.batch_size = cfg.CONST.BATCH_SIZE
        
        self.img_w = cfg.CONST.IMG_W
        self.img_h = cfg.CONST.IMG_H
        self.n_vox = cfg.CONST.N_VOX

        # (self.batch_size, 3, self.img_h, self.img_w),
        # override x and is_x_tensor4 when using multi-view network
        self.is_x_tensor4 = True

    def parameter_init(self):
        #initialize all the parameters of the gru net
        if hasattr(self, "encoder") and hasattr(self, "decoder"):
            for m in self.modules():
                
                if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                    """
                    For Conv2d, the shape of the weight is 
                    (out_channels, in_channels, kernel_size[0], kernel_size[1]).
                    For Conv3d, the shape of the weight is 
                    (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]).
                    """
                    w_shape = (m.out_channels, m.in_channels, *m.kernel_size)
                    m.weight.data = weight_init(w_shape)
                    if m.bias is not None:
                        m.bias.data.fill_(0.1)
                        
                elif isinstance(m, nn.Linear):
                    """
                    For Linear module, the shape of the weight is (out_features, in_features)
                    """
                    w_shape = (m.out_features, m.in_features)
                    m.weight.data = weight_init(w_shape)
                    if m.bias is not None:
                        m.bias.data.fill_(0.1)
        else:
            raise Exception("The network must have an encoder and a decoder before initializing all the parameters")
                
    def forward(self, x, y=None):
        raise NotImplementedError("Define a forward pass")
