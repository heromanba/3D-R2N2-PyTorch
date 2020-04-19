import os
import sys
from datetime import datetime

from lib.config import cfg
from lib.utils import Timer, has_nan

import torch
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable


def max_or_nan(params):
    params = list(params)
    nan_or_max_param = torch.FloatTensor(len(params)).zero_()
    if torch.cuda.is_available():
        nan_or_max_param = nan_or_max_param.cuda()
        
    for param_idx, param in enumerate(params):
        # If there is nan, max will return nan
        # Note that param is Variable
        nan_or_max_param[param_idx] = torch.max(torch.abs(param)).item()
        # print('param %d : %f' % (param_idx, nan_or_max_param[param_idx]))
    return nan_or_max_param


class Solver(object):

    def __init__(self, net):
        self.net = net
        self.lr = cfg.TRAIN.DEFAULT_LEARNING_RATE
        print('Set the learning rate to %f.' % self.lr)
        
        #set the optimizer
        self.set_optimizer(cfg.TRAIN.POLICY)

    def set_optimizer(self, policy=cfg.TRAIN.POLICY):
        """
        This function is used to set the optimization algorithm of training 
        """
        net = self.net
        lr = self.lr
        w_decay = cfg.TRAIN.WEIGHT_DECAY
        if policy == 'sgd':
            momentum = cfg.TRAIN.MOMENTUM
            self.optimizer = SGD(net.parameters(), lr=lr, weight_decay=w_decay, momentum=momentum)
        elif policy == 'adam':
            self.optimizer = Adam(net.parameters(), lr=lr, weight_decay=w_decay)
        else:
            sys.exit('Error: Unimplemented optimization policy')

    def train_loss(self, x, y):
        """
        y is provided and test is False, only the loss will be returned.

        Args:
            x (torch.Tensor): batch_img_tensor
            y (torch.Tensor): batch_voxel_tensor
        """
        
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            
        loss = self.net(x, y, test=False)
        
        #compute gradient and do parameter update step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

    def train(self, train_loader, val_loader=None):
        ''' Given data queues, train the network '''
        # Parameter directory
        save_dir = os.path.join(cfg.DIR.OUT_PATH)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Timer for the training op and parallel data loading op.
        train_timer = Timer()
        data_timer = Timer()
        training_losses = []

        # Setup learning rates
        lr_steps = [int(k) for k in cfg.TRAIN.LEARNING_RATES.keys()]
        
        #Setup the lr_scheduler
        self.lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer, lr_steps, gamma=0.1)

        start_iter = 0
        # Resume training
        if cfg.TRAIN.RESUME_TRAIN:
            self.load(cfg.CONST.WEIGHTS)
            start_iter = cfg.TRAIN.INITIAL_ITERATION

        # Main training loop
        train_loader_iter = iter(train_loader)
        for train_ind in range(start_iter, cfg.TRAIN.NUM_ITERATION + 1):
            self.lr_scheduler.step()
            
            data_timer.tic()
            try:
                batch_img, batch_voxel = train_loader_iter.next()
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch_img, batch_voxel = train_loader_iter.next()

            data_timer.toc()

            if self.net.is_x_tensor4:
                batch_img = batch_img[0]

            # Apply one gradient step
            train_timer.tic()
            loss = self.train_loss(batch_img, batch_voxel)
            train_timer.toc()

            training_losses.append(loss.item())

            # Decrease learning rate at certain points
            if train_ind in lr_steps:
                #for pytorch optimizer, learning rate can only be set when the optimizer is created
                #or using torch.optim.lr_scheduler
                print('Learing rate decreased to %f: ' % cfg.TRAIN.LEARNING_RATES[str(train_ind)])

            # Debugging modules
            #
            # Print status, run validation, check divergence, and save model.
            if train_ind % cfg.TRAIN.PRINT_FREQ == 0:
                # Print the current loss
                print('%s Iter: %d Loss: %f' % (datetime.now(), train_ind, loss))

            if train_ind % cfg.TRAIN.VALIDATION_FREQ == 0 and val_loader is not None:
                # Print test loss and params to check convergence every N iterations

                val_losses = 0
                val_num_iter = min(cfg.TRAIN.NUM_VALIDATION_ITERATIONS, len(val_loader))
                val_loader_iter = iter(val_loader)
                for i in range(val_num_iter):
                    batch_img, batch_voxel = val_loader_iter.next()
                    val_loss = self.train_loss(batch_img, batch_voxel)
                    val_losses += val_loss
                var_losses_mean = val_losses / val_num_iter
                print('%s Test loss: %f' % (datetime.now(), var_losses_mean))

            if train_ind % cfg.TRAIN.NAN_CHECK_FREQ == 0:
                # Check that the network parameters are all valid
                nan_or_max_param = max_or_nan(self.net.parameters())
                if has_nan(nan_or_max_param):
                    print('NAN detected')
                    break

            if (train_ind % cfg.TRAIN.SAVE_FREQ == 0 and not train_ind == 0) or \
                    (train_ind == cfg.TRAIN.NUM_ITERATION):
                # Save the checkpoint every a few iterations or at the end.
                self.save(training_losses, save_dir, train_ind)

            #loss is a Variable containing torch.FloatTensor of size 1
            if loss.item() > cfg.TRAIN.LOSS_LIMIT:
                print("Cost exceeds the threshold. Stop training")
                break

    def save(self, training_losses, save_dir, step):
        ''' Save the current network parameters to the save_dir and make a
        symlink to the latest param so that the training function can easily
        load the latest model'''

        save_path = os.path.join(save_dir, 'checkpoint.%d.pth' % (step))
        
        #both states of the network and the optimizer need to be saved
        state_dict = {'net_state': self.net.state_dict()}
        state_dict.update({'optimizer_state': self.optimizer.state_dict()})
        torch.save(state_dict, save_path)
        
        # Make a symlink for weights.npy
        symlink_path = os.path.join(save_dir, 'checkpoint.pth')
        if os.path.lexists(symlink_path):
            os.remove(symlink_path)
            
        # Make a symlink to the latest network params
        os.symlink("%s" % os.path.abspath(save_path), symlink_path)

        # Write the losses
        with open(os.path.join(save_dir, 'loss.%d.txt' % step), 'w') as f:
            f.write('\n'.join([str(l) for l in training_losses]))


    def load(self, filename):
        if os.path.isfile(filename):
            print("loading checkpoint from '{}'".format(filename))
            if torch.cuda.is_available():
                checkpoint = torch.load(filename)
            else:
                checkpoint = torch.load(filename, map_location='cpu')
            
            net_state = checkpoint['net_state']
            optim_state = checkpoint['optimizer_state']
            
            self.net.load_state_dict(net_state)
            self.optimizer.load_state_dict(optim_state)
        else:
            raise Exception("no checkpoint found at '{}'".format(filename))
    

    def test_output(self, x, y=None):
        """
        Generate the reconstruction, loss, and activation. Evaluate loss if
        ground truth output is given. Otherwise, return reconstruction and
        activation.
        In test mode, if y is None, then the out is the [prediction].
        In test mode, if y is not None, then the out is [prediction, loss].

        Args:
            x (torch.Tensor): batch_img_tensor
            y (torch.Tensor): batch_voxel_tensor
        """
        
        if torch.cuda.is_available():
            x = x.cuda()
            if y is not None:
                y = y.cuda()
        
        # Parse the result
        results = self.net(x, y, test=True)
        prediction = results[0]
        loss = results[1]
        activations = results[2:]
        if y is None:
            return prediction, activations
        else:
            return prediction, loss, activations
