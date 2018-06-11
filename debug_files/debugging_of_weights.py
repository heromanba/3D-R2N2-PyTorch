from models import load_model
from demo import DEFAULT_WEIGHTS
import theano
from lib.solver import Solver
if __name__ == '__main__':
    
    theano.config.floatX = 'float32'
    net_class = load_model('ResidualGRUNet')
    net = net_class()
    net.load(DEFAULT_WEIGHTS)
    
    solver = Solver(net)
    
    """
    import scipy.io as sio 
    a = sio.loadmat("/Users/wangchu/Desktop/PASCAL3D+_release1.0 2/CAD/aeroplane.mat")
    
    from lib.read_mesh import parse_mtl, file_exists
    a = parse_mtl("/Users/wangchu/Desktop/PASCAL3D+_release1.0 2/CAD/aeroplane.mat")
    """