import torch
import numpy as np
from sklearn.cluster import KMeans
from numpy.linalg import eig

def run_kmeans(H, n_clusters):
    H_np = H.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(H_np)
    return kmeans.labels_, kmeans.cluster_centers_

def compute_sw(H, centers, labels):
    H_centered = H - centers[labels]

    scatter_matrix = sum(H_centered[labels == i].T @ H_centered[labels == i]
                         for i in range(len(centers)))
    
    return scatter_matrix

def compute_eigen(scatter_matrix):
    eigvals, eigvecs = eig(scatterr_matrix)
    idx = np.argsort(eigvals)[::-1]

    return eigvecs[:,idx]