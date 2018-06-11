## Branch description
```
This branch contains codes of the PyTorch version of 3D-R2N2.
```

## Citing this work

If you find this work useful in your research, please consider citing:

```
@inproceedings{choy20163d,
  title={3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction},
  author={Choy, Christopher B and Xu, Danfei and Gwak, JunYoung and Chen, Kevin and Savarese, Silvio},
  booktitle = {Proceedings of the European Conference on Computer Vision ({ECCV})},
  year={2016}
}
```

## Project Page

The project page is available at [http://cvgl.stanford.edu/3d-r2n2/](http://cvgl.stanford.edu/3d-r2n2/).


## Datasets

We used [ShapeNet](http://shapenet.cs.stanford.edu) models to generate rendered images and voxelized models which are available below (you can follow the installation instruction below to extract it to the default directory).

- ShapeNet rendered images [ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz](ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz)
- ShapeNet voxelized models [ftp://cs.stanford.edu/cs/cvgl/ShapeNetVox32.tgz](ftp://cs.stanford.edu/cs/cvgl/ShapeNetVox32.tgz)
- Trained ResidualGRUNet Weights [ftp://cs.stanford.edu/cs/cvgl/ResidualGRUNet.npy](ftp://cs.stanford.edu/cs/cvgl/ResidualGRUNet.npy)


## Installation原始版(论文github给的安装)

The package requires python3. You can follow the direction below to install virtual environment within the repository or install anaconda for python 3.

- Download the repository

```
git clone https://github.com/chrischoy/3D-R2N2.git
```

- Setup virtual environment and install requirements and copy the theanorc file to the `$HOME` directory

```
cd 3D-R2N2
pip install virtualenv
virtualenv -p python3 --system-site-packages py3
source py3/bin/activate
pip install -r requirements.txt
cp .theanorc ~/.theanorc
```

## Installation实际版(我用学校的gpu时的安装)

学校装好了Nvidia driver和cuda

- 查看gpu状态
```
nvidia-smi
```
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.90                 Driver Version: 384.90                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:03:00.0 Off |                  N/A |
| 50%   83C    P2    89W / 250W |   1329MiB / 11172MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```
- 查看cuda版本
```
nvcc --version
```
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Tue_Jan_10_13:22:03_CST_2017
Cuda compilation tools, release 8.0, V8.0.61
```
- 安装cudnn(来自[installation guide](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-deb)但是cuda和cudnn版本不一样)
```
cd <cudnnpath>
```
Install the runtime library
```
sudo dpkg -i libcudnn7_7.1.1.5-1+cuda8.0_amd64.deb
```
Install the developer library
```
sudo dpkg -i libcudnn7-dev_7.1.1.5-1+cuda8.0_amd64.deb
```
Install the code samples and the cuDNN Library User Guide
```
sudo dpkg -i libcudnn7-doc_7.1.1.5-1+cuda8.0_amd64.deb
```
- Verify cudnn(和它网站上的一样)

http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#verify

- 安装anaconda
```
bash Anaconda3-5.1.0-Linux-x86_64.sh
```
- 用conda安装libpygpu(theano gpu backend)

这个gpu backend好像是老的，新的是pygpu，在theano的conda-forge里装的有
```
conda install -c anaconda libgpuarray
```
- 用conda安装theano
```
conda install -c conda-forge theano 
```
- 用theano可能有的MKL2018 bug
```
RuntimeError: To use MKL 2018 with Theano you MUST set "MKL_THREADING_LAYER=GNU" in your environement.
```
把下面一行加到~/.bashrc中
```
export MKL_THREADING_LAYER=GNU
```
- 测试theano能不能用gpu和cudnn(来自[testing theano with gpu](http://deeplearning.net/software/theano/tutorial/using_gpu.html#testing-theano-with-gpu))

拷贝链接中的代码，并命名为testing_theano_with_gpu.py
```
THEANO_FLAGS='device=cuda0, floatX=float32, force_device=True, optimizer_including=cudnn' python3 testing_theano_with_gpu.py 
```
结果如下
```
Using cuDNN version 7101 on context None
Disabling allocation cache on cuda0
Mapped name None to device cuda0: GeForce GTX 1080 Ti (0000:03:00.0)
[GpuElemwise{exp,no_inplace}(<GpuArrayType<None>(float32, vector)>), HostFromGpu(gpuarray)(GpuElemwise{exp,no_inplace}.0)]
Looping 1000 times took 0.445036 seconds
Result is [1.2317803 1.6187935 1.5227807 ... 2.2077181 2.2996776 1.623233 ]
Used the gpu
```
- 用pip安装easydict

anaconda有requirements.txt里要求的大部分库，除了easydict。但是用conda装easydict会和conda里的其他库冲突，所以用pip安装
```
pip install easydict
```
- Thanks my GD

### Running demo.py

- Install meshlab (skip if you have another mesh viewer). If you skip this step, demo code will not visualize the final prediction.

```
sudo apt-get install meshlab
```

- Run the demo code and save the final 3D reconstruction to a mesh file named `prediction.obj`

```
python demo.py prediction.obj
```

The demo code takes 3 images of the same chair and generates the following reconstruction.

| Image 1         | Image 2         | Image 3         | Reconstruction                                                                            |
|:---------------:|:---------------:|:---------------:|:-----------------------------------------------------------------------------------------:|
| ![](imgs/0.png) | ![](imgs/1.png) | ![](imgs/2.png) | <img src="https://github.com/chrischoy/3D-R2N2/blob/master/imgs/pred.png" height="127px"> |

- Deactivate your environment when you are done

```
deactivate
```


### Training the network

- Activate the virtual environment before you run the experiments.

```
source py3/bin/activate
```

- Download datasets and place them in a folder named `ShapeNet`

```
mkdir ShapeNet/
wget ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz
wget ftp://cs.stanford.edu/cs/cvgl/ShapeNetVox32.tgz
tar -xzf ShapeNetRendering.tgz -C ShapeNet/
tar -xzf ShapeNetVox32.tgz -C ShapeNet/
```

- Train and test the network using the training shell script

```
./experiments/scripts/res_gru_net.sh
```

**Note**: The initial compilation might take awhile if you run the theano for the first time due to various compilations. The problem will not persist for the subsequent runs.

### 训练时可能会有的"gpu out of memory" bug
```
pygpu.gpuarray.GpuArrayException: Out of memory
```
解决方法(来自https://groups.google.com/d/msg/theano-users/kpfsem4tabA/bjOuVtCIAwAJ)

1.在~/.theanorc中设置
```
[gpuarray]
preallocate = -1
```
2.在所有processes spawn后再标注使用gpu
```
详见lib/train_net.py第70、71行和lib/test_net.py第45、46行
```
## License

MIT License
