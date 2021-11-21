# Annotation of Spatially Resolved Single-cell Data with STELLAR

This repo contains the reference source code in PyTorch for STELLAR.

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