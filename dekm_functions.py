import torch
import numpy as np
from sklearn.cluster import KMeans
from numpy.linalg import eig

def run_kmeans(H, n_clusters):
    H_np = H.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(H_np)
    return kmeans.labels_, kmeans.cluster_centers_
