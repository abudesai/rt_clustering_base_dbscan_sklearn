DBSCAN Model build in Sklearn for Clustering - Base problem category as per Ready Tensor specifications.

- DBSCAN
- clustering
- sklearn
- python
- pandas
- numpy
- docker

This is a Clustering Model that uses DBSCAN implemented through Sklearn.

Density-based spatial clustering of applications with noise (DBSCAN) is a density-based clustering non-parametric algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away).

The data preprocessing step includes:

- for numerical variables
  - Standard scale data

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as iris, penguins, landsat_satellite, geture_phase_classification, vehicle_silhouettes, spambase, steel_plate_fault. Additionally, we also used synthetically generated datasets such as two concentric (noisy) circles, and unequal variance gaussian blobs.

This Clustering Model is written using Python as its programming language. ScikitLearn is used to implement the main algorithm, evaluate the model, and preprocess the data. Numpy, pandas, Sklearn, and feature-engine are used for the data preprocessing steps.

There are no web endpoints provided for this model. Training and prediction is performed by issuing command `train_predict` on the docker container. Also see usage in the `run_local.py` file inside `local_test` directory.
