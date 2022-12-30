import streamlit as st
import numpy as np
import requests
import time
from json.encoder import JSONEncoder


st.set_page_config(
    page_title="Diamond price estimator",
    page_icon=":diamond:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# This is a baseline diamond price estimator, based on Kaggle data found at [https://www.kaggle.com/datasets/swatikhedekar/price-prediction-of-diamond](https://www.kaggle.com/datasets/swatikhedekar/price-prediction-of-diamond)"
    }
)

st.title('Diamond price estimator')

#Select box values
cut = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']


url = 'http://127.0.0.1:5000'
try:
    status = requests.get(url)
except:
    st.text(f'API connexion impossible.')
    st.stop()
    
if status.status_code != 200:
    st.text(f'API connexion impossible. Returned status: {status.status_code}')

else:
    with st.form("diamond_data", clear_on_submit=False):
        st.header("Enter the diamond's main characteristics below")

        diamond_char={}

        col1, col2, col3 = st.columns(3)
        with col1:
            diamond_char['carat'] = st.number_input(label="Enter the diamond's weight in carats", min_value=0.00)
            diamond_char['cut'] = st.selectbox(label="Select the diamond's cut", options=cut)
            diamond_char['color'] = st.selectbox(label="Select the diamond's color", options=color)
        with col2:
            diamond_char['clarity'] = st.selectbox(label="Select the diamond's clarity", options=clarity)
            diamond_char['table'] = st.number_input(label="Enter the diamond's table", help="The table is the width of top of diamond relative to widest point")
        with col3:
            diamond_char['x'] = st.number_input(label="Enter the diamond's length in mm", min_value=0.00)
            diamond_char['y'] = st.number_input(label="Enter the diamond's width in mm", min_value=0.00)
            diamond_char['z'] = st.number_input(label="Enter the diamond's depth in mm", min_value=0.00)

        submitted = st.form_submit_button("Submit diamond information")
        
    if submitted:
        if (diamond_char['x']==0)|(diamond_char['y']==0)|(diamond_char['z']==0)|(diamond_char['carat']==0):
            st.text("Please check the values for weight, length, width, and depth and re-submit the diamond information.")
        else:
            st.subheader('Diamond price estimate')
            
            progress_status = st.empty()
            with progress_status.container():
                with st.spinner(text='Estimation in progress'):
                    diamond_char['depth'] = 2*diamond_char['z']/(diamond_char['x']+diamond_char['y'])
                    diamond_char_json = JSONEncoder().encode(diamond_char)
                    diam_pricing = requests.post(url=url+'/upload', json=diamond_char_json, verify=False)
                    diam_pricing = diam_pricing.json()
                st.success('Estimation complete')
                time.sleep(1)
            progress_status.empty()

            st.markdown(f"<span style='font-size:50px'>The diamond is estimated at USD <span style='color:#52be80'>**{diam_pricing['model_output']*diamond_char['carat']:,.2f}**</span></span>.",
                        unsafe_allow_html=True)
            st.write(f"With 94% probability, the price will be between USD {diam_pricing['min_response']*diamond_char['carat']:,.2f} and USD {diam_pricing['max_response']*diamond_char['carat']:,.2f}.")
