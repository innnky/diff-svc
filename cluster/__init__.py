import numpy as np
import torch
from sklearn.cluster import KMeans

kmeans = KMeans(100)
checkpoint = torch.load("cluster/kmeans_100.pt")
kmeans.__dict__["n_features_in_"] = checkpoint["n_features_in_"]
kmeans.__dict__["_n_threads"] = checkpoint["_n_threads"]
kmeans.__dict__["cluster_centers_"] = checkpoint["cluster_centers_"]

def get_cluster_result(x):
    """x: np.array [t, 256]"""
    return kmeans.predict(x)
