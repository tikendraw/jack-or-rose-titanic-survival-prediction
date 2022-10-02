import streamlit as st
from PIL import Image
import pandas as pd
import joblib
import webbrowser
import os
# os.system('pip install emoji --quiet')
from emoji import emojize

st.title('Would you have survived TITANIC ?')


dataset_url = 'https://www.kaggle.com/competitions/titanic/data'
notebook_url = 'https://github.com/tikendraw/Would_you_have_survived_TITANIC/blob/main/titanic.ipynb'
#link buttons
col11, col21, col31 = st.columns([1,1,1])

with col11:
    if st.button('Dataset'):
        webbrowser.open_new_tab(dataset_url)
with col21:
    if st.button('Code/Notebook'):
        webbrowser.open_new_tab(notebook_url)
with col31:
    st.button('Reload')


# Buttons
name = st.text_input('Name')
sex = (st.radio("Sex", ('Male', 'Female'))).lower()
age = st.number_input('Age', max_value=100, min_value=0, step = 1, value = 24)

col1, col2, col3 = st.columns([1,1,1])

with col1:
    pclass = st.selectbox('PClass', [1,2,3])

with col2:
    sibsp = st.number_input('Number of Siblings/Spouse', min_value= 0 , max_value=10, step = 1)

with col3:
    parch = st.number_input('Number of Parents/Childern', min_value= 0 , max_value=10, step = 1)


col4, col5 = st.columns([2,1])

with col4:
    embarked = st.selectbox('Depart Station',['Southampton(S)', 'Cherbourg(C)', 'Queenstown(Q)'])

with col5:
    fare = st.number_input('Select Ticket price', min_value= 0 , max_value=500, step = 1, value= 25 )


embark_station = {
                'Southampton(S)':'S', 
                'Cherbourg(C)':'C', 
                'Queenstown(Q)':'Q'}

input_data = pd.DataFrame([[pclass, age, sibsp, parch, fare, sex, embarked]],
columns=['pclass', 'age', 'sibsp','parch','fare','sex','embarked'])

#mapping embark station codes
input_data['embarked'] = input_data['embarked'].map(embark_station)


def preprocess(data):
    #encoding sex
    sex_ohe = joblib.load('./saved/sex_ohe.joblib')
    sex_enc = sex_ohe.transform(data['sex'].to_numpy().reshape(-1,1))
    sex_enc = pd.DataFrame(sex_enc, columns=[sex_ohe.categories_[0][-1]])

    #encoding embark
    embark_ohe = joblib.load('./saved/embark_ohe.joblib')
    embark_enc = embark_ohe.transform(data['embarked'].to_numpy().reshape(-1,1))
    embark_enc = pd.DataFrame(embark_enc, columns=[*embark_ohe.categories_[0]])

    # dropping categories
    data.drop(['embarked','sex'], axis  = 1, inplace=True)

    # joining encoded embark
    data = pd.concat([data, sex_enc, embark_enc], axis = 1)

    # scaler
    scaler = joblib.load('./saved/big_scaler.joblib')
    data = scaler.transform(data)

    return data

#loading model
model = joblib.load('./model/XGBClassifier.joblib')

#load images
rose_img = Image.open('./static/rose1.webp')
jack_img = Image.open('./static/jack.gif')


# loading some gifs
import base64
file_ = open("./static/jack.gif", "rb")
contents = file_.read()
jack_url = base64.b64encode(contents).decode("utf-8")
file_.close()

file_ = open("./static/rose1.webp", "rb")
contents = file_.read()
rose_url = base64.b64encode(contents).decode("utf-8")
file_.close()


if st.button('Predict'):
    data =  preprocess(input_data)  
    result = model.predict(data)

    if result > .5:
        good = f"{name}! you will Live."
        st.markdown(f'<p style="color:#2AC153;font-size:24px;border-radius:2%;"><b>{good}</b></p>', unsafe_allow_html=True)
        st.markdown(f'<img src="data:image/gif;base64,{rose_url}" alt="cat gif">',
                        unsafe_allow_html=True)
    else :
        bad = f"{name}! you are dead(jacked). \U0001F4A6 \U0001F976 \U0001F480"
        st.markdown(f'<p style="color:#E44D32;font-size:24px;border-radius:2%;"><b>{bad}</b></p>', unsafe_allow_html=True)
        st.markdown(f'<img src="data:image/gif;base64,{jack_url}" alt="cat gif">',
                        unsafe_allow_html=True)







