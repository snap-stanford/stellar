# Annotation of Spatially Resolved Single-cell Data with STELLAR

PyTorch implementation of STELLAR, a geometric deep learning tool for cell-type discovery and identification in spatially resolved single-cell datasets. For a detailed description of the algorithm, please see our manuscript [Annotation of Spatially Resolved Single-cell Data with STELLAR](https://www.biorxiv.org/content/10.1101/2021.11.24.469947v1.full.pdf).


<p align="center">
<img src="https://github.com/snap-stanford/stellar/blob/main/images/stellar_overview.png" width="1100" align="center">
</p>



### Dependencies
- [PyTorch](https://pytorch.org/)
- [PyG](https://pytorch-geometric.readthedocs.io/en/latest/)
- [sklearn](https://scikit-learn.org/)

### Getting started

We implemented STELLAR model in a self-contained class. To make an instance and train STELLAR:

```
stellar = STELLAR(args, labeled_X, labeled_y, unlabeled_X, labeled_pos, unlabeled_pos)
stellar.train()
_, results = stellar.pred()
```

## Datasets

CODEX multiplexed imaging datasets from healthy human tonsil and Barrett’s esophagus data are made available at [dryad](https://datadryad.org/stash/share/1OQtxew0Unh3iAdP-ELew-ctwuPTBz6Oy8uuyxqliZk).


## Citing

If you find our code and research useful, please consider citing:

```
@article{stellar2021,
  title={Annotation of Spatially Resolved Single-cell Data with STELLAR},
  author={Brbic, Maria and Cao, Kaidi and Hickey, John W and Tan, Yuqi and Snyder, Michael and Nolan, Garry P and Leskovec, Jure},
  journal={bioRxiv},
  year={2021},
}
```