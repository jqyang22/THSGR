
<div align="center">

<h1>Boosting Multimodal Remote Sensing Image Classification with Transformer-based Heterogeneously Salient Graph Representation</h1>

<h2>IEEE TGRS 2026</h2>


[Jiaqi Yang](https://jqyang22.github.io/)<sup>a</sup>, [Bo Du](https://cs.whu.edu.cn/info/1019/2892.htm/)<sup>b</sup>, [Rong Liu](https://gp.sysu.edu.cn/teacher/3702/)<sup>c</sup>, [Zhu Mao](https://www.helsinki.fi/en/about-us/people/people-finder/zhu-mao-9492290)<sup>d</sup>, [Liangpei Zhang](https://www.zhangliangpei.cn/)<sup>b</sup>

<sup>a</sup> University of Wisconsin-Madison,
<sup>b</sup> Wuhan University,
<sup>c</sup> Sun Yat-sen University, 
<sup>d</sup> University of Helsinki.

</div>

<div align="center">

<p align='center'>
  <a href="https://ieeexplore-ieee-org.ezproxy.library.wisc.edu/document/11494135"><img alt="Pape" src="https://img.shields.io/badge/TGRS-Paper-6D4AFF?style=for-the-badge" /></a>
</p>

<p align="center">
  <a href="#-overview">Overview</a> |
  <a href="#-project-structure">Project Structure</a> |
  <a href="#-datasets">Datasets</a> |
  <a href="#-installation">Installation</a> |
  <a href="#-usage">Usage</a> |
  <a href="#-citation">Citation</a>
</p>


</div>


# 🧩 Overview

Data collected by different modalities can provide a wealth of complementary information, such as hyperspectral image (HSI) to offer rich spectral-spatial properties, synthetic aperture radar (SAR) to provide structural information about the Earth's surface, and light detection and ranging (LiDAR) to cover altitude information about ground elevation. Therefore, a natural idea is to combine multimodal images for refined and accurate land-cover interpretation. Although many efforts have been attempted to achieve multi-source remote sensing image classification, there are still three issues as follows: 1) indiscriminate feature representation without sufficiently considering modal heterogeneity, 2) abundant features and complex computations associated with modeling long-range dependencies, and 3) overfitting phenomenon caused by sparsely labeled samples. To overcome the above barriers, a transformer-based heterogeneously salient graph representation (THSGR) approach is proposed in this paper. First, a multimodal heterogeneous graph encoder is presented to encode distinctively non-Euclidean structural features from heterogeneous data. Then, a self-attention-free multi-convolutional modulator is designed for effective and efficient long-term dependency modeling. Finally, a mean forward strategy is developed in order to avoid overfitting. Based on the above structures, the proposed model is able to break through modal gaps to obtain differentiated graph representation with competitive time cost, even for a small fraction of training samples. Experiments and analyses in three benchmark datasets with various state-of-the-art (SOTA) approaches show the performance of the proposed THSGR.</a>


<figure>
<div align="center">
<img src=Fig/THSGR.bmp width="80%">
</div>

<div align='center'>
 
**Figure 1. Flowchart of THSGR.**

</div>

<div align='center'>

</div>
<br>

# 📁 Project Structure

```
THSGR/
├── main.py                       # full pipeline
├── config/
│   └── config.yaml               # dataset / network / output configuration
├── loadData/
│   ├── data_pipe.py              # DataLoader assembly
│   ├── data_reader.py            # raw .mat readers
│   └── split_data.py             # train / val / test split utilities
└── models/
    ├── THSGR.py                   # main THSGR network
    └── transformer.py            # transformer structure
```

# 🌍 Datasets
HSI-SAR Augsburg: https://github.com/danfenghong/ISPRS_S2FL <br>
HSI-LiDAR Houston: https://machinelearning.ee.uh.edu/2013-ieee-grss-data-fusion-contest/ <br>
HSI-SAR Berlin: https://github.com/danfenghong/ISPRS_S2FL <br>

<div align='center'>
</div>


# 📦 Installation

This project is implemented with **PyTorch**:

| Package | Version |
| :------ | :------ |
| pytorch | 1.7+ |
| numpy | 1.21.4 |
| matplotlib | 3.3.3 |
| scikit-learn | 0.23.2 |
| einops | 0.4+ |
| timm | 0.6+ |
| thop | 0.1+ |
| pyyaml | 5.4+ |
| scipy | 1.7+ |
| pandas | 1.3+ |

Install with:

```bash
conda create -n thsgr python=3.8 -y
conda activate thsgr
pip install torch==1.7.1 torchvision==0.8.2
pip install numpy==1.21.4 matplotlib==3.3.3 scikit-learn==0.23.2 \
            einops timm thop pyyaml scipy pandas xlwt
```

# 🔨 Usage
## 1. Prepare data
Download the HSI / LiDAR `.mat` files and place them under one folder.

## 2. Configure paths
Edit `config/config.yaml` for the data path.

## 3. Train & test

```bash
python main.py --device cuda:0  --path-config /your/path/THSGR/config/config.yaml
```

# ⭐ Citation

If you find this work helpful, please give a ⭐ and cite it as follows:

```
@ARTICLE{11494135,
  author={Yang, Jiaqi and Du, Bo and Liu, Rong and Mao, Zhu and Zhang, Liangpei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Boosting Multimodal Remote Sensing Image Classification with Transformer-based Heterogeneously Salient Graph Representation}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Earth Observing System;Sentinel-1;Sentinel-2;Apertures;Feeds;Antennas;Filtering;Filters;Modulation;Communications technology;Multimodal classification;HSI-SAR/LiDAR imagery;heterogeneously salient graph representation;transformer},
  doi={10.1109/TGRS.2026.3686762}}
```
