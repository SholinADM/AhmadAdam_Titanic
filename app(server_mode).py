#IMPORT PACKAGES
import numpy as np
import pandas as pd
import mlflow
import streamlit as st
from PIL import Image
import os
import pickle
import json
import requests
serving_host ="http://127.0.0.1:5001"
#%%
st.title('Titanic Grim Reaper')
st.header("WE ARE ABOUT TO SINK!!!", divider=True)
st.subheader('Why struggle when you got no hope of surviving? Or do you? With the power of statistics, fate is now only a probability theory.')
st.write('Input me the information below to know your fate.')
#CREATE A FORM FOR USER INPUT
with st.form(key='my_form'):
    Pclass = st.slider("Passenger's Class", 1, 3, value = 1,step = 1)
    Sex = st.selectbox("Sex",('male', 'female'))
    Age = st.number_input("Age",0.0,100.0)
    SibSp = st.slider("Number of Sibling/Spouse onboard", 0,8,value = 1,step = 1)
    Parch = st.slider("Number of Parent/children onboard", 0,6,value = 1,step = 1)
    Fare = st.number_input("Fare's price",0.0,100.0)
    Embarked = st.selectbox("Embarked gate", ('S', 'C', 'Q'))
    submit = st.form_submit_button("Submit")
st.write("hit submit to input to the inference server")
#dictionary to encode object inputs
enc = {"male":1.0,
       "female":0.0,
       "C":0.0,
       "Q":1.0,
       "S":2.0}

#format user input into dictionary (the select option default to zero)
dict ={'Pclass' : Pclass,
        'Sex' : enc[Sex],
        'Age' : Age,
        'SibSp' : SibSp,
        'Parch' : Parch,
        'Fare'  : Fare,
        'Embarked' : enc[Embarked]}
#Convert dictionary into a dataframe
df = pd.DataFrame(dict, index=[0])

#make a prediction
payload = json.dumps({"dataframe_split": df.to_dict(orient="split")})
outcome = requests.post(
    url=f"{serving_host}/invocations",
    data=payload,
    headers={"Content-Type": "application/json"},
)
if submit:
    outcome=outcome.json()["predictions"][0]["0"]
    outcome= np.round(outcome*100,2)
    st.subheader(f"Your probability of Surviving is : {outcome}%")
    if outcome > 80:
        st.success("WHAT ARE YOU DOING HERE?? GET TO THE EMMERGENCY RAFT RIGHT AWAY!")
    elif outcome > 30:
        st.warning("Best have luck on you, you might survive")
    elif outcome >= 0:
        st.error("Cry and Pray, you're going down with this ship.")
    else:
        st.error("somethig went wrong")