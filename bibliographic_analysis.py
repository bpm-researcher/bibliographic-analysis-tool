import streamlit as st

import network_analysis
import performance_analysis
import qualitative_analysis
import science_mapping

st.set_page_config(page_title="Bibliographic Analysis", layout="wide")

st.title("Bibliographic Analysis")

tabs = st.tabs(["Performance Analysis", "Network Analysis", "Science Mapping", "Quantitative Analysis - Models"])


with tabs[0]:
    performance_analysis.show()

with tabs[1]:
    network_analysis.show()

with tabs[2]:
    science_mapping.show()

with tabs[3]:
    qualitative_analysis.show()


