import streamlit as st
import os

#from utils import setup_sidebar #list_available_dates,create_date_selectbox,update_dataframes
import plotly.express as px


st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
)

st.write("# Claim Count Forecast: a Long Term Model! 👋")

st.sidebar.success("Select a report above.")

st.markdown(
    """
    We provide you with a monthly review of the Claim Count Forecast.
    **👈 Select a report from the sidebar** to start
    ### Need help?
    - Ping us on Teams
"""
)

