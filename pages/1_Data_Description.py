"""
Web app: data description functions.
"""

import streamlit as st
import numpy as np
import pandas as pd

import download as dl
import util
from figures import two_feature_barplot, state_geomap

st.write(
    """
    ### Data Description
    **Select data from the preprocessed dataset.**
    """
)

plot_opt = ["Feature distribution over another feature (stacked barplot)",
            "Feature distribution per municipality (geospatial map)"]

preprocessed_data = dl.get_SIM()
selected_states = st.multiselect("States: ", options=util.available_states, default=util.available_states)
selected_years = st.multiselect("Years: ", options=util.available_years, default=util.available_years)
selected_data = preprocessed_data.loc[(preprocessed_data['state'].isin(selected_states)) & 
                                      (preprocessed_data['year'].isin(selected_years))]
categorical_features = selected_data.select_dtypes(exclude=[np.number]).columns.tolist()
categorical_features.append("year")

if st.button(label="Describe", type="primary"):
    st.write("Descriptive statistics for numerical features: ", selected_data.describe(include=np.number),
             "Descriptive statistics for categorical features: ", selected_data.describe(include=['O']))

st.write("**Visualize selected data.**")

plot = st.selectbox("Select a visualization option: ", options=plot_opt)

if plot == plot_opt[0]:
    # "Feature distribution per year (barplot)"
    plot_feature = st.selectbox("Plot feature: ", options=categorical_features)
    axis_feature = st.selectbox("Axis feature: ", options=categorical_features)
    percent_y = st.checkbox("Percentage y-axis?")
    #age_group = st.selectbox("Age group: ", options=preprocessed_data['age_group'].unique())
    if st.button(label="Plot", type="primary"):
        st.pyplot(two_feature_barplot(selected_data, plot_feature, axis_feature, percent_y))

elif plot == plot_opt[1]:
    # "Feature distribution per municipality"
    st.write("This option will display a map for each selected state.")
    feature = st.selectbox("Feature: ", options=preprocessed_data.columns)
    if st.button(label="Plot", type="primary"):
        for state in selected_states:
            st.pyplot(state_geomap(preprocessed_data, state, feature))
