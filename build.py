
import math
import numpy as np
import pickle
import streamlit as st
import os
import warnings
warnings.filterwarnings("ignore")
# from librosa import librosa
# SET PAGE WIDE
st.set_page_config(page_title='Music Genre Prediction', layout="centered", initial_sidebar_state="collapsed",

                   page_icon="icon.jpeg",
                   menu_items={
                       'Get Help': 'https://github.com/regnna',
                       'Report a bug': 'https://github.com/regnna',
                       'About': 'Regnna'
                   })


st.markdown("<h1 style='text-align: center; color: gold;'> Find Your Music Genres </h1>",
            unsafe_allow_html=True)

hide_st_style = """
                <style>
                header{visibility:hidden;}
                footer{visibility:hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)
# Add background image

st.markdown(
    f"""
         <style>
         .stApp {{
             background-image:
              url("https://images.unsplash.com/photo-1470225620780-dba8ba36b745?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
    unsafe_allow_html=True
)

model,scaler=pickle.load(open('gbregression.pkl','rb'))
