# Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation (PCS)

Pytorch implementation of PCS (Prototypical Cross-domain Self-supervised network) [[PDF](_)]

## Overview

Architecture of Network

![Architecture of Network](./images/framework.png)

Compared with state-of-the-art methods, PCS improves the mean classification accuracy over different domain pairs on FUDA by **10.5%**, **4.3%**, **9.0%**, and **13.2%** on Office, Office-Home, VisDA-2017, and DomainNet, respectively.
q
## Requirements

pytorch==1.5.1
torchvision==0.6.1
cuda version 10.2

```bash
pip install -r requirements.txt
pip install -e .
```

## Training

- Download or soft-link your dataset under `data` folder (Split files are provided in `data/splits`, supported datasets are Office, Office-Home, VisDA-2017, and DomainNet)
- To train the model, run following commands:

```bash
CUDA_VISIBLE_DEVICES=0,1 python pcs/run.py --config config/office_D-A.json --num 1
CUDA_VISIBLE_DEVICES=2,3 python pcs/run.py --config config/office_A-W.json --num 3
```

## Citation

```bibtex
@InProceedings{Yue_2021_Prototypical,
author = {Yue, Xiangyu and Zheng, Zangwei and Zhang, Shanghang and Gao, Yang and Darrell, Trevor and Keutzer, Kurt and Sangiovanni-Vincentelli, Alberto},
title = {Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021}
}
```

## Acknowlegdement

This code is built on [[MME](https://github.com/VisionLearningGroup/SSDA_MME)]. We thank the authors for sharing their codes.
