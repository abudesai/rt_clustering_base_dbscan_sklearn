from dataclasses import replace
import dis
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances

warnings.filterwarnings("ignore")


model_fname = "model.save"

MODEL_NAME = "clustering_base_dbscan"


class ClusteringModel:
    def __init__(self, eps, min_samples, **kwargs) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.model = self.build_model()

    def build_model(self):
        model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
        )
        return model

    def fit_predict(self, *args, **kwargs):
        return self.model.fit_predict(*args, **kwargs)

    def evaluate(self, x_test):
        """Evaluate the model and return the loss and metrics"""
        raise NotImplementedError

    def save(self, model_path):
        joblib.dump(self, os.path.join(model_path, model_fname))

    @classmethod
    def load(cls, model_path):
        clusterer = joblib.lo


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):
    model = ClusteringModel.load(model_path)
    return model


def get_data_based_model_params(data):
    """
    Set any model parameters that are data dependent.
    For example, number of layers or neurons in a neural network as a function of data shape.
    """
    # https://stats.stackexchange.com/questions/88872/a-routine-to-choose-eps-and-minpts-for-dbscan
    min_samples = data.shape[1]

    eps = get_percentile_distance(data)
    return {"min_samples": min_samples, "eps": eps}


def get_percentile_distance(data):
    N = data.shape[0]
    num_samples = min(100, N)
    samples = data.sample(n=num_samples, replace=False, axis=0)
    distances = euclidean_distances(samples, samples).flatten()
    distances = sorted(distances)[num_samples:]  # exclude self distances
    perc_value = np.percentile(
        distances, 2.0
    )  # percentile on distance - this needs to be hyper-parameter tuned
    return perc_value
