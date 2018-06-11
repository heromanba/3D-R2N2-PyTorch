#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 17:08:33 2018

@author: wangchu
"""
import theano
   
#replace original operations in LeakyReLU.set_output with a function
#which will be replaced by ModifiedBackprop
def leakyReLU_func(self_input, self_leakiness=0.01):
    if self_leakiness:
        # The following is faster than T.maximum(leakiness * x, x),
        # and it works with nonsymbolic inputs as well. Also see:
        # http://github.com/benanne/Lasagne/pull/163#issuecomment-81765117
        f1 = 0.5 * (1 + self_leakiness)
        f2 = 0.5 * (1 - self_leakiness)
        self_output = f1 * self_input + f2 * abs(self_input)
        # self.param = [self.leakiness]
    else:
        self_output = 0.5 * (self_input + abs(self_input))
    return self_output


"""utility class for visualization"""
class ModifiedBackprop(object):
    
    def __init__(self, nonlinearity):
        self.nonlinearity = nonlinearity
        self.ops = {}   #memorize an OpFromGraph instance per tensor type
        
    def __call__(self, x):
        """
        #OpFromGraph is oblique to Theano optimizations, so we need to move thins
        #to gpu ourselves if needed
        if theano.sandbox.cuda.cuda_enabled:
            maybe_to_gpu = theano.sandbox.cuda.as_cuda_ndarray_variable
        else:
            maybe_to_gpu = lambda x: x
        #we move the input to gpu if needed
        x = maybe_to_gpu(x)
        """
        
        #we note the tensor type of the input variable to the nonlinearity
        #(mainly dimensionality and dtype); we need to create a fitting Op.
        tensor_type = x.type
        #if we did not create a suitable Op yet, this is the time to do so.
        if tensor_type not in self.ops:
            #For the graph, we create an input variable of the correct type:
            inp = tensor_type()
            #we pass it through the nonlinearity (and move to gpu if needed)
            outp = self.nonlinearity(inp)
            #then we fix the forward expression
            op = theano.OpFromGraph([inp], [outp])
            #and replace the gradient with out own (defined in a subclass)
            op.grad = self.grad
            #finally, we memorize the new op
            self.ops[tensor_type] = op
        #and apply the memorize op to the input we got
        return self.ops[tensor_type](x)
    
class GuidedBackprop(ModifiedBackprop):
    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        dtype = inp.dtype
        return (grd * (inp > 0).astype(dtype) * (grd > 0).astype(dtype),)



