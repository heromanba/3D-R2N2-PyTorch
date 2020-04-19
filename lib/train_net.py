
import inspect
from multiprocessing import Queue

from torch.utils.data import DataLoader

# Training related functions
from models import load_model
from lib.config import cfg
from lib.dataset import ShapeNetDataset, ShapeNetCollateFn
from lib.solver import Solver
from lib.data_io import category_model_id_pair
from lib.data_process import kill_processes, make_data_processes

# Define globally accessible queues, will be used for clean exit when force
# interrupted.


def cleanup_handle(func):
    '''Cleanup the data processes before exiting the program'''

    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            print('Wait until the dataprocesses to end')
            # kill_processes(train_queue, train_processes)
            # kill_processes(val_queue, val_processes)
            raise

    return func_wrapper


@cleanup_handle
def train_net():
    '''Main training function'''
    # Set up the model and the solver
    NetClass = load_model(cfg.CONST.NETWORK_CLASS)

    net = NetClass()
    print('\nNetwork definition: ')
    print(net)

    # Check that single view reconstruction net is not used for multi view
    # reconstruction.
    if net.is_x_tensor4 and cfg.CONST.N_VIEWS > 1:
        raise ValueError('Do not set the config.CONST.N_VIEWS > 1 when using' \
                         'single-view reconstruction network')

    # Prefetching data processes
    #
    # Create worker and data queue for data processing. For training data, use
    # multiple processes to speed up the loading. For validation data, use 1
    # since the queue will be popped every TRAIN.NUM_VALIDATION_ITERATIONS.

    train_dataset = ShapeNetDataset(cfg.TRAIN.DATASET_PORTION)
    train_collate_fn = ShapeNetCollateFn()
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.CONST.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKER,
        collate_fn=train_collate_fn,
        pin_memory=True
    )

    val_dataset = ShapeNetDataset(cfg.TEST.DATASET_PORTION)
    val_collate_fn = ShapeNetCollateFn(train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.CONST.BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        collate_fn=val_collate_fn,
        pin_memory=True
    )

    net.cuda()

    # Generate the solver
    solver = Solver(net)

    # Train the network
    solver.train(train_loader, val_loader)


def main():
    '''Test function'''
    cfg.DATASET = '/cvgl/group/ShapeNet/ShapeNetCore.v1/cat1000.json'
    cfg.CONST.RECNET = 'rec_net'
    cfg.TRAIN.DATASET_PORTION = [0, 0.8]
    train_net()


if __name__ == '__main__':
    main()
