import numpy as np
import collections
import torch

#utility function to check nan
def has_nan(x):
    """
    x is a torch tensor. (x != x) will return a torch.ByteTensor whose 
    elements are either 1 or 0. And (x != x).any() will return True if
    any elements in the tensor are non-zero. Note that (nan != nan) is 
    True. If there is any nan in x, then the function will return True.
    """
    return (x != x).any()

#utility function to customize weight initialization
def weight_init(w_shape, mean=0, std=0.01, filler='msra'):
    rng = np.random.RandomState()
    if isinstance(w_shape, collections.Iterable):
        if len(w_shape) > 1 and len(w_shape) < 5:
            fan_in = np.prod(w_shape[1:])
            fan_out = np.prod(w_shape) / w_shape[1]
            n = (fan_in + fan_out) / 2.
        elif len(w_shape) == 5:
            # 3D Convolution filter
            fan_in = np.prod(w_shape[1:])
            fan_out = np.prod(w_shape) / w_shape[2]
            n = (fan_in + fan_out) / 2.
        else:
            raise NotImplementedError(
                    'Filter shape with ndim > 5 not supported: len(w_shape) = %d' % len(w_shape))
    else:
        raise Exception("w_shape should be an instance of collections.Iterable")
    
    if filler == 'gaussian':
        np_values = np.asarray(rng.normal(mean, std, w_shape))
    elif filler == 'msra':
        np_values = np.asarray(rng.normal(mean, np.sqrt(2. / n), w_shape))
    elif filler == 'xavier':
        scale = np.sqrt(3. / n)
        np_values = np.asarray(rng.uniform(low=-scale, high=scale, size=w_shape))
    elif filler == 'constant':
        np_values = mean * np.ones(w_shape)
    elif filler == 'orth':
        ndim = np.prod(w_shape)
        W = np.random.randn(ndim, ndim)
        u, s, v = np.linalg.svd(W)
        np_values = u.reshape(w_shape)
    else:
        raise NotImplementedError('Filler %s not implemented' % filler)
    torch_tensor = torch.from_numpy(np_values).type(torch.FloatTensor)
    return torch_tensor

###############################################################################
#                                                                             #
#                  original time utility class                                #
#                                                                             #
###############################################################################

import time

class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
