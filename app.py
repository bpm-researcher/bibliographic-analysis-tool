import streamlit as st

from data.data_preparation import show as show_data
from analysis.performance_analysis import show as show_performance
from analysis.science_mapping import show as show_science_mapping

st.set_page_config(page_title="Bibliographic Analysis", layout="wide")

st.title("Bibliographic Analysis")

tabs = st.tabs(["Data Preparation", "Performance Analysis", "Science Mapping"])

with tabs[0]:
    show_data()

with tabs[1]:
    show_performance()

with tabs[2]:
    show_science_mapping()





