
import os
import numpy as np
import scipy.io as sio
import inspect
import sklearn.metrics
from multiprocessing import Queue

from torch.utils.data import DataLoader

# Theano & network
from models import load_model
from lib.config import cfg
from lib.dataset import ShapeNetDataset, ShapeNetCollateFn
from lib.solver import Solver
from lib.data_io import category_model_id_pair
from lib.data_process import make_data_processes, get_while_running

from lib.voxel import evaluate_voxel_prediction, voxel2obj


def test_net():
    ''' Evaluate the network '''
    # Make result directory and the result file.
    result_dir = os.path.join(cfg.DIR.OUT_PATH, cfg.TEST.EXP_NAME)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_fn = os.path.join(result_dir, 'result.mat')

    print("Exp file will be written to: " + result_fn)

    # Make a network and load weights
    NetworkClass = load_model(cfg.CONST.NETWORK_CLASS)

    #print('Network definition: \n')
    #print(inspect.getsource(NetworkClass.network_definition))

    net = NetworkClass() 
    net.cuda()

    net.eval()
    
    solver = Solver(net)
    solver.load(cfg.CONST.WEIGHTS)

    # set constants
    batch_size = cfg.CONST.BATCH_SIZE

    # set up testing data process. We make only one prefetching process. The
    # process will return one batch at a time.

    test_dataset = ShapeNetDataset(dataset_portion=cfg.TEST.DATASET_PORTION)
    test_collate_fn = ShapeNetCollateFn(train=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=test_collate_fn,
        pin_memory=True
    )

    num_data = len(test_dataset)
    num_batch = int(num_data / batch_size)

    # prepare result container
    results = {'cost': np.zeros(num_batch),
               'mAP': np.zeros((num_batch, batch_size))}
    # Save results for various thresholds
    for thresh in cfg.TEST.VOXEL_THRESH:
        results[str(thresh)] = np.zeros((num_batch, batch_size, 5))

    # Get all test data
    batch_idx = 0
    for batch_img, batch_voxel in test_loader:
        if batch_idx == num_batch:
            break

        #activations is a list of torch.cuda.FloatTensor
        pred, loss, activations = solver.test_output(batch_img, batch_voxel)
        
        #convert pytorch tensor to numpy array
        pred = pred.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()
        batch_voxel_np = batch_voxel.cpu().numpy()

        for j in range(batch_size):
            # Save IoU per thresh
            for i, thresh in enumerate(cfg.TEST.VOXEL_THRESH):
                r = evaluate_voxel_prediction(pred[j, ...], batch_voxel_np[j, ...], thresh)
                results[str(thresh)][batch_idx, j, :] = r

            # Compute AP
            precision = sklearn.metrics.average_precision_score(
                batch_voxel[j, 1].flatten(), pred[j, 1].flatten())

            results['mAP'][batch_idx, j] = precision

        # record result for the batch
        results['cost'][batch_idx] = float(loss)
        print('%d/%d, costs: %f, mAP: %f' %
                (batch_idx, num_batch, loss, np.mean(results['mAP'][batch_idx])))
        batch_idx += 1


    print('Total loss: %f' % np.mean(results['cost']))
    print('Total mAP: %f' % np.mean(results['mAP']))

    # sio.savemat(result_fn, results)
