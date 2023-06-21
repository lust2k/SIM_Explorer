'''
Functions for cluster analysis.
'''

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score   

from util import impute_df
from figures import plot_silhouette
    
def evaluate_clustering(df, linkage_matrix, dist_values, gen_plots=False) -> tuple:
    """
    Evaluate clustering results for a list of distance threshold values.
    """
    results = []
    plots = []
    for dist in dist_values:
        labels = fcluster(linkage_matrix, t=dist, criterion='distance')
        n_clusters = len(np.unique(labels))
        sil = silhouette_score(df.values, labels)
        ch = calinski_harabasz_score(df.values, labels)
        results.append((dist, n_clusters, sil, ch))
        #results.append(f"Parameter = {dist}: Silhouette score = {sil}; CH score = {ch}; Number of clusters = {n_clusters}")
        if gen_plots:
            plots.append(plot_silhouette(df.values, labels))
    results = pd.DataFrame(results, columns=['Parameter', 'Clusters (k)', 'Silhouette score', 'CH score'])
    return (results, plots)

def apply_linkage(df, selected_method='complete') -> tuple:
    """
    Apply clustering algorithm. Return one-hot encoded data and the linkage matrix.
    """
    # Impute missing data
    df = impute_df(df)
    # Apply one-hot encoding to categorical features
    categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_features, dtype=float)
    # Get the linkage matrix
    dist_matrix = pdist(df)
    linkage_matrix = linkage(dist_matrix, method=selected_method)
    return (df, linkage_matrix)

def apply_labels(df, linkage_matrix, dist: int) -> pd.DataFrame:
    labels = fcluster(linkage_matrix, t=dist, criterion='distance')
    df['cluster'] = labels
    df.to_csv('./data/labeled_data.csv', index=False)
    return df