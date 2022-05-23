# Annotation of Spatially Resolved Single-cell Data with STELLAR

PyTorch implementation of STELLAR, a geometric deep learning tool for cell-type discovery and identification in spatially resolved single-cell datasets. For a detailed description of the algorithm, please see our manuscript [Annotation of Spatially Resolved Single-cell Data with STELLAR](https://www.biorxiv.org/content/10.1101/2021.11.24.469947v1.full.pdf).


<p align="center">
<img src="https://github.com/snap-stanford/stellar/blob/main/images/stellar_overview.png" width="1100" align="center">
</p>



### Installation

**Requirements**

- NVIDIA GPU, Linux, Python3. We test our software on Ubuntu 16.04 with NVIDIA Geforce 2080 Ti GPU and 1T CPU memory. 


**1. Python environment (Optional):**
We recommend using Conda package manager

```bash
conda create -n stellar python=3.8
source activate stellar
```

**2. Pytorch:**
Install [PyTorch](https://pytorch.org/). 
We have verified under PyTorch 1.9.1. For example:
```bash
conda install pytorch cudatoolkit=11.3 -c pytorch
```

**3. Pytorch Geometric:**
Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), 
follow their instructions. We have verified under Pyg 2.0. For example:
```bash
conda install pyg -c pyg
```

**4. Other dependencies:**

Please run the following command to install additional packages.
```bash
pip install -r requirements.txt
```

### Getting started

We implemented STELLAR model in a self-contained class. To make an instance and train STELLAR:

```
stellar = STELLAR(args, dataset)
stellar.train()
_, results = stellar.pred()
```
### Datasets

CODEX multiplexed imaging datasets from healthy human tonsil and Barrett’s esophagus data are made available at [dryad](https://datadryad.org/stash/share/1OQtxew0Unh3iAdP-ELew-ctwuPTBz6Oy8uuyxqliZk). Our demo code assumes the data to be put under the folder `./data/ `.

### Demo

We provide several training examples with this repo:

- To run STELLAR on the CODEX healthy intestine data

```bash
python STELLAR_run.py --dataset Hubmap --input-dim 48 --num-heads 22
```

- To run STELLAR on the CODEX tonsil and BE data:

```bash
python STELLAR_run.py --dataset TonsilBE --input-dim 44 --num-heads 13 --num-seed-class 3
```

We also provided a jupyter notebook ``demo.ipynb`` that walks through a downsampled dataset. Please consider downsample more if there's still a memory issue. Note that the performance of the model would degrade as the training data gets less. For users with limited memory and potentially limited access to GPU, we create another notebook that loads pre-processsed data and can finish with CPU in about 30 mins.

### Use your own dataset

Our stellar function requires node features, corresponding labels and corresponding edges as inputs. Here Node feature matrix should have shape [num_nodes, num_node_features] and edge indexes should have shape [2, num_edges].

### Citing

If you find our code and research useful, please consider citing:

```
@article{stellar2021,
  title={Annotation of Spatially Resolved Single-cell Data with STELLAR},
  author={Brbic, Maria and Cao, Kaidi and Hickey, John W and Tan, Yuqi and Snyder, Michael and Nolan, Garry P and Leskovec, Jure},
  journal={bioRxiv},
  year={2021},
}
```