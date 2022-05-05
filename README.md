# Annotation of Spatially Resolved Single-cell Data with STELLAR

PyTorch implementation of STELLAR, a geometric deep learning tool for cell-type discovery and identification in spatially resolved single-cell datasets. For a detailed description of the algorithm, please see our manuscript [Annotation of Spatially Resolved Single-cell Data with STELLAR](https://www.biorxiv.org/content/10.1101/2021.11.24.469947v1.full.pdf).


<p align="center">
<img src="https://github.com/snap-stanford/stellar/blob/main/images/stellar_overview.png" width="1100" align="center">
</p>



### Dependencies

STELLAR requires the following packages. We test our software on Ubuntu 16.04 with NVIDIA Geforce 2080 Ti GPU. Please check the [requirements.txt](https://github.com/snap-stanford/stellar/blob/main/requirements.txt) file for more details on required Python packages. 

- [PyTorch==1.9](https://pytorch.org/)
- [PyG==1.7](https://pytorch-geometric.readthedocs.io/en/latest/)
- [sklearn==1.0.1](https://scikit-learn.org/)

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

### Use your own dataset

Please refer to `load_hubmap_data()` and implement your own loader and construct the dataset.

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