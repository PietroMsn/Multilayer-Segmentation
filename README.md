# Multilayer Segmentation (extension of Pointcept framework)

This is an extension of the Pointcept framework for multilayer clothed human segmentation. The Backbones implemented for this comparison are: Pointnet++, DGCNN and PointTransformer. See the multialyer segmentation paper for more details [Paper] (https://github.com/PietroMsn/Scanned-Gim3D).
**Pointcept**is a powerful and flexible codebase for point cloud perception research. It is also an official implementation of the following paper:

- **Pointnet++**
*Qi, Charles Ruizhongtai, Li Yi, Hao Su, Leonidas J. Guibas*
"Pointnet++: Deep hierarchical feature learning on point sets in a metric space."
Advances in neural information processing systems 30 (2017).

- **DGCNN**
*Phan, Anh Viet, Minh Le Nguyen, Yen Lam Hoang Nguyen, Lam Thu Bui*
"Dgcnn: A convolutional neural network over large-scale labeled graphs." 
Neural Networks 108 (2018): 533-543.

- **Point Transformer**   
*Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip Torr, Vladlen Koltun*  
IEEE International Conference on Computer Vision (**ICCV**) 2021 - Oral  



## Citation
If you find _Pointcept_ useful to your research, please cite our work as encouragement.
```
@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished = {\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}
```

## Overview

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Model Zoo](#model-zoo)
- [Acknowledgement](#acknowledgement)

## Installation

### Requirements
- Ubuntu: 18.04 and above.
- CUDA: 11.3 and above.
- PyTorch: 1.10.0 and above.

### Conda Environment
- **Method 1**: Utilize conda `environment.yml` to create a new environment with one line code:
  ```bash
  # Create and activate conda environment named as 'pointcept-torch2.5.0-cu12.4'
  # cuda: 12.4, pytorch: 2.5.0

  # run `unset CUDA_PATH` if you have installed cuda in your local environment
  conda env create -f environment.yml --verbose
  conda activate pointcept-torch2.5.0-cu12.4
  ```

- **Method 2**: Manually create a conda environment:
  ```bash
  conda create -n pointcept python=3.10 -y
  conda activate pointcept
  
  # (Optional) If no CUDA installed
  conda install nvidia/label/cuda-12.4.1::cuda conda-forge::cudnn conda-forge::gcc=13.2 conda-forge::gxx=13.2 -y
  
  conda install ninja -y
  # Choose version you want here: https://pytorch.org/get-started/previous-versions/
  conda install pytorch==2.5.0 torchvision==0.13.1 torchaudio==0.20.0 pytorch-cuda=12.4 -c pytorch -y
  conda install h5py pyyaml -c anaconda -y
  conda install sharedarray tensorboard tensorboardx wandb yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
  conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
  pip install torch-geometric

  # spconv (SparseUNet)
  # refer https://github.com/traveller59/spconv
  pip install spconv-cu124

  # PPT (clip)
  pip install ftfy regex tqdm
  pip install git+https://github.com/openai/CLIP.git

  # PTv1 & PTv2 or precise eval
  cd libs/pointops
  # usual
  python setup.py install
  # docker & multi GPU arch
  TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
  # e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
  TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
  cd ../..

  # Open3D (visualization, optional)
  pip install open3d
  ```

## Data Preparation - Scanned GIM3D

Here the scenned GIM3D dataset can be donwloaded: [Scanned GIM3D Download](https://univr-my.sharepoint.com/:f:/g/personal/pietro_musoni_univr_it/EjZ0-1KtCn1NhqmtcdkuDngBK5l3gXg5kc-_8AZ3N0sBZA?e=AU90nM).
All the folders contained in the root must be copied in the directory ```data``` of the current code folder.


## Quick Start

### Training

For launching the trainings for all the strategies descripted in the paper use the following batch script:
```
for i in {1..5}; do sh scripts/train.sh -g 1 -d gim3d -c pt-t$i-aug -n pt-t$i-aug; done
```

Alternatively, each training can be launched independently.
Here for example the code for the first strategy:
```
sh scripts/train.sh -g 1 -d gim3d -c pt-t1-aug -n pt-t1-aug
```
In order to change the configuration (e.g. the barch size), you must edit the file: ```configs/gim3d/pt-t*-aug.py ```


### Testing

For launching the test for the scanned GIM3D dataset you must launch the batch file:
```
sh scripts/test.sh -g 1 -d gim3d -c pt-t1-aug -n pt-t1-aug
```

## Acknowledgement
_Pointcept_ is designed by [Xiaoyang](https://xywu.me/), named by [Yixing](https://github.com/yxlao) and the logo is created by [Yuechen](https://julianjuaner.github.io/). It is derived from [Hengshuang](https://hszhao.github.io/)'s [Semseg](https://github.com/hszhao/semseg) and inspirited by several repos, e.g., [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [pointnet2](https://github.com/charlesq34/pointnet2), [mmcv](https://github.com/open-mmlab/mmcv/tree/master/mmcv), and [Detectron2](https://github.com/facebookresearch/detectron2).
