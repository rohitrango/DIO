# Deep Implicit Optimization for Robust and Flexible Image Registration

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2111.10480-b31b1b.svg)](https://arxiv.org/abs/2111.10480)

keywords: image registration, optimization

This is a **PyTorch** implementation of our paper:

<a href="https://arxiv.org/abs/2406.07361">Jena, Rohit, et.al. Deep Implicit Optimization for Robust and Flexible Image Registration</a>

More coming soon.


## Installation instructions

Run the commands as 

```PYTHONPATH=./ python ... ```

## Brain

```python
python train_multi_level_3d.py --config-name oasis_4x2x1x_unetencoder exp_name=oasis_lku_l2 loss.img_loss=mse train.epochs=500 model.name=lku dataset.data_root=/path/to/OASIS/
```

## Lung

```python
PYTHONPATH=./ python scripts/dio/train_multi_level_3d_kps.py --config-name nlst exp_name=nlst_lkumini_fireants_alllvl_tv10.0 diffopt.warp_type=diffeomorphic diffopt.learning_rate=0.5 train.train_new_level=[0] loss.weight_tv=10.0
```
