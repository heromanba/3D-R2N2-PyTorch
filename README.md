# 3D-R2N2-PyTorch
This is a pytorch version of 3D-R2N2. Original repo: https://github.com/chrischoy/3D-R2N2

## Installation
The repo was tested with python3.6, cuda 10.1, pytorch 1.4.0. You can follow the instruction below to install the virtual environment.

- Get the source code.
```bash
git clone https://github.com/heromanba/3D-R2N2-PyTorch.git
```

- Install anaconda(https://docs.anaconda.com/anaconda/install/).

- Create virtual environment and install required packages.
```bash
cd 3D-R2N2-PyTorch
conda create -n 3D-R2N2 python=3.6
conda activate 3D-R2N2
pip install -r requirements.txt
```

## Demo
- Download pretrained model(ResidualGRUNet), and put ```checkpoint.pth``` under ```output/ResidualGRUNet/default_model```.

    Google drive link(https://drive.google.com/open?id=1LtNhuUQdAeAyIUiuCavofBpjw26Ag6DP)

    Baidu pan link(链接: https://pan.baidu.com/s/12YK4mnQNx9xdCjzV7zx7GA 提取码: 66nf)

- Run
The predicted object will be saved to ```prediction.obj```.
```bash
python demo.py
```

## Train
### Prepare dataset
- Use the same dataset as mentioned in the original repo.

    ShapeNet rendered images http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz

    ShapeNet voxelized models http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz

- Extract data into ```ShapeNet``` directory, the file structure in ```ShapeNet``` should be like this:
```
ShapeNet/
    |
    |----ShapeNetRendering/
    |
    |----ShapeNetVox32/
    |
```

- Change some parameters. You can change parameters in ```experiments/scripts/res_gru_net.sh``` or ```lib/config.py```

- Run.
```bash
bash experiments/scripts/res_gru_net.sh
```

## License
MIT License
