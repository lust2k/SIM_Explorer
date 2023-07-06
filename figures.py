"""
Functions for data visualization.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geobr
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score, silhouette_samples   

from unidecode import unidecode

import util

available_states = util.available_states
available_years = util.available_years

def plot_dendrogram(linkage_matrix, levels: int) -> plt.figure:
    """
    Generate dendrogram.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(linkage_matrix, truncate_mode='level', p=levels)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Distance')
    return fig

def plot_silhouette(df, labels) -> plt.figure:
    """
    Generate silhouette plot.
    """
    silhouette_avg = silhouette_score(df, labels)
    silhouette_vals = silhouette_samples(df, labels)
    y_lower, y_upper = 0, 0
    fig, ax = plt.subplots()
    for i, cluster in enumerate(set(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        color = plt.cm.Set1(i / len(set(labels)))
        ax.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1.0,
                edgecolor='none', color=color)
        y_lower += len(cluster_silhouette_vals)
        
    ax.axvline(silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    return fig

def two_feature_barplot(df: pd.DataFrame, plot_feature: str, axis_feature: str, percent_y: bool = False):
    """
    Generate a barplot of the distribution of a feature over another feature.
    """
    # Group the data by year and method
    df = df.groupby([axis_feature, plot_feature]).size().unstack(fill_value=0)
    if percent_y:
        df = df.apply(lambda x: x / x.sum() * 100, axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot(kind='bar', stacked=True, legend=True, ax=ax)
    ax.set_xlabel(f"{axis_feature}")
    ax.set_title(f"Distribution of {plot_feature} over {axis_feature}")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig

def feature_cluster_heatmap(df, feature: str) -> plt.figure:
    """
    Generate heatmap for feature distribution per cluster.
    """
    features = ["cluster"]
    # Select all columns related to the feature (handle one-hot encoded features)
    for column in df.columns:
        if feature in column:
            features.append(column)
            
    # Create a new dataframe with the cluster labels and the selected feature
    df = df.loc[:, features]
    means = df.groupby('cluster').mean()
    fig, ax = plt.subplots()
    ax = sns.heatmap(means, cmap='coolwarm', annot=True, fmt='.2f')
    ax.set_title("Means of %s per cluster" % feature)
    return fig

def state_geomap(df: pd.DataFrame, state, feature: str) -> plt.figure:
    """
    Generate a map of the state with distribution of a feature per municipality
    """
    state_map = geobr.read_municipality(code_muni=state)
    #state_map['code_muni'] = (state_map['code_muni'] / 10).astype(int)
    #df['CODMUN'] = df['CODMUN'].astype(int)
    #state_map = state_map.merge(df, how='left', left_on='code_muni', right_on='CODMUN')
    state_map['name_muni'] = state_map['name_muni'].str.lower()
    df['name_muni'] = df['name_muni'].str.lower()
    state_map = state_map.merge(df, how='left', on='name_muni')
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    state_map.plot(
        column=feature,
        cmap="Blues",
        edgecolor="#FEBF57",
        legend=True,
        legend_kwds={
            "label": "Number of suicides per 100k inhabitants",
            "orientation": "vertical",
            "shrink": 0.4,
        },
        ax=ax,
    )
    ax.set_title(f"Distribution of {feature} in {state}", fontsize=20)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.axis("off")
    return fig
