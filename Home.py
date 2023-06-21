"""
Streamlit web application.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="SIM analytics")

st.markdown(
    """
    ### About
    This app is a tool to analyze suicide data in Brazil. 
    It provides a preprocessed mortality database and functions to visualize the data, as well as a simple framework to apply clustering algorithms. 
    """
)
