"""
Web app: cluster analysis functions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re

import download as dl
import clustering as cl
import util
from figures import plot_dendrogram, feature_cluster_heatmap, state_geomap

st.write(
    """
    ### Cluster Analysis
    **Select data, apply the desired hierarchical clustering method and visualize the dendrogram.**
    """
)

preprocessed_data = dl.get_SIM()
selected_states = st.multiselect("States: ", options=util.available_states, default=util.available_states)
selected_years = st.multiselect("Years: ", options=util.available_years, default=util.available_years)
default_columns = ['IDADE', 'LOCOCOR', 'SEXO', 'RACACOR', 'ESC', 'ESTCIV', 'age_group', 'method', 'season', 'day_period', 'weekday', 'facility_rate']
selected_feats = st.multiselect("Features: ", options=preprocessed_data.columns.to_list(), default=default_columns)
selected_method = st.selectbox("Clustering method: ", options=['Single (nearest point)',
                                                      'Complete (farthest point)',
                                                      'Average (UPGMA)',
                                                      'Weighted (WPGMA)',
                                                      'Centroid (UPGMC)',
                                                      'Median (WPGMC)',
                                                      'Ward']).split()[0].lower()
selected_data = preprocessed_data[selected_feats].loc[(preprocessed_data['state'].isin(selected_states)) & 
                                                      (preprocessed_data['year'].isin(selected_years))]

if st.button(label="Plot Dendrogram", type='primary'):
    (onehot_data, linkage_matrix) = cl.apply_linkage(selected_data, selected_method)
    dendro = plot_dendrogram(linkage_matrix, levels=4)
    st.pyplot(dendro)

# --------

st.write(
    """
    **Evaluate cluster quality.**\n
    Provide a list of distance threshold values to test. The dendrogram y-axis can give insights on these values.
    """
)

list_input = st.text_input("Insert a list of values separated by commas:")
gen_plots = st.checkbox("Generate silhouette coefficient plots?")

collect_numbers = lambda x : [int(i) for i in re.split("[^0-9]", x) if i != ""]
dist_values = collect_numbers(list_input)
if st.button(label="Evaluate", type='primary'):
    (onehot_data, linkage_matrix) = cl.apply_linkage(selected_data, selected_method)
    (results, plots) = cl.evaluate_clustering(onehot_data, linkage_matrix, dist_values, gen_plots)
    st.write(results)
    if gen_plots:
        for plot in plots:
            st.pyplot(plot)

st.write("""
    **Set the distance threshold and label the data.**\n
    A column named "cluster" will be added to the dataset.
    """
)

labeled_data = pd.read_csv('./data/labeled_data.csv')
dist_threshold = st.number_input(label="Distance threshold:", min_value=1, step=1)
if st.button(label="Apply", type='primary'):
    (onehot_data, linkage_matrix) = cl.apply_linkage(selected_data, selected_method)
    labeled_data = cl.apply_labels(onehot_data, linkage_matrix, dist_threshold)

st.write(
    """
    Current clusters:
    """,
    labeled_data['cluster'].value_counts()
)

categorical_features = labeled_data.select_dtypes(exclude=[np.number]).columns.tolist()

plot_opt = ["Feature distribution per cluster (heatmap)",
            "Geographic distribution of clusters (geospatial map)"]

plot = st.selectbox("Select a plot to visualize: ", options=plot_opt)
if plot == plot_opt[0]:
    feature = st.selectbox("Feature:", options=selected_feats)
    if st.button(key="heatmap", label="Plot", type="primary"):
        st.pyplot(feature_cluster_heatmap(labeled_data, feature))
elif plot == plot_opt[1]:
    state = st.selectbox("Feature:", options=categorical_features)
    if st.button(key="geomap", label="Plot", type="primary"):
        st.pyplot(state_geomap(labeled_data, state))
