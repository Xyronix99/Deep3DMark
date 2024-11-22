# Deep3DMark
Official implementation of paper "[Rethinking Mesh Watermark: Towards Highly Robust and Adaptable Deep 3D Mesh Watermarking](https://arxiv.org/abs/2307.11628)". AAAI 2024. Xingyu Zhu, Guanhui Ye, Xiapu Luo, Xuetao Wei.

## Enviroment

```
conda create -n Deep3DMark python=3.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

# install local cuda env
cd model/backend
pip install .
```



## Data Preparation

Deep3DMark is trained on [*m2500*](https://drive.google.com/file/d/1MG1sO2khPBc77hyxD-O3lvH9qP7lXqDO/view?usp=drive_link) (decimated ModelNet40 dataset with target vertex number=2500) and *m500* (decimated ModelNet40 dataset with target vertex number=500) and evaluted on the entire [ModelNet40](https://modelnet.cs.princeton.edu/). We preprocess ModelNet40 using [CGAL](https://doc.cgal.org/latest/Surface_mesh_simplification/index.html) to obtain decimated dataset.

With *m2500* installed, you need to update dataset root directory in the provided config file ```config/debug_wm.yaml```, ```config/GAT8_2_2_1.yaml```:
```
train_set:
  ...
  root: # update path to m2500 root directory
  ...

valid_set:
  ...
  root: # update path to m2500 root directory
  ...
```


## Running Models

### Evalutaion
We provide [ckpts](https://drive.google.com/file/d/1Jk-rfOobD0aczSb32zneHtJxX1AuIagl/view?usp=drive_link) of both pretrained msg encoder/decoder and the Deep3DMark.

With ckpt, ModelNet40 and *m2500* installed. You can run 
```
python debug.py --config config/debug_wm.yaml
```
to reproduce **SNR** & **robustness** & **size adapatation** results reported in paper.

### Training
To train a Deep3DMark model, run:
```
python train.py --config config/GAT8_2_2_1.yaml
```


### Citation
```
@inproceedings{zhu2024rethinking,
  title={Rethinking Mesh Watermark: Towards Highly Robust and Adaptable Deep 3D Mesh Watermarking},
  author={Zhu, Xingyu and Ye, Guanhui and Luo, Xiapu and Wei, Xuetao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={7784--7792},
  year={2024}
}
```