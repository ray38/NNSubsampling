# Nearest Neighbour Based Subsampling Algorithm

This is a sampling algorithm aiming to achieve near-uniform sampling of high dimentional data sets

The sampling algorithm is an iterative process based on Nearest Neighbour search. 

In each iteration, the dataset is normalized with the standard scaler (mean = 0, standard deviation = 1) and a nearest-neighbour model is constructed and queried to find the nearest neighbor for each data point as well as the distance between them. If the distance is below a certain cutoff distance, the neighbor is removed with some probability. The process is iterated until there are no more points to be removed. 

The algorithm has two hyper-parameters: cutoff distance and deletion probability (rate). 

The cutoff distance controls the sparsity of the resulting representative dataset. Higher cutoff distances resulting in fewer sub-sampled points.
The deletion probability controls robustness. Lower deletion probablibity is more robust but resulting in slower execution. High deletion probablity might result in a "hole" in the subsampled dataset

High-dimensional datasets may also be pre-processed by principal component analysis (PCA) to reduce the dimensionality prior to subsampling


### For more detailed explanation of the code and some tests, please find the jupyter notebook "NNSubsampling_Explaination_and_Test.ipynb" under "./Tutorial"


# Dependencies

* Python 2 or Python 3

* scikit-

* [pykdtree](https://github.com/storpipfugl/pykdtree) (optional, recommended)

* [nmslib](https://github.com/nmslib/nmslib/tree/master/python_bindings) (optional, recommended)

* [FLANN](http://www.cs.ubc.ca/research/flann/) (optional)

* [Annoy](https://github.com/spotify/annoy) (optional)

* scipy (optional)
