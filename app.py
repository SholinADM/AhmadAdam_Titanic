#IMPORT PACKAGES
import numpy as np
import pandas as pd
import mlflow
import streamlit as st
from PIL import Image
import os
import pickle

@st.cache_resource
def load_model(MODEL_URI):
    model = mlflow.keras.load_model(model_uri = MODEL_URI)
    return model

@st.cache_resource
def load_encoder(run_id):
    artifact_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/encoders/encoder.pkl")
    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)
    return artifact

#model URI and run ID constants, in comment are stable version
#"models:/titanic_reaper/1" 
#"b558234262fe470da5cf70b8c01e5782"
MODEL_URI ="models:/titanic_reaper/3"
run_id = "e30f1a1161bb47539dcbd1778841e05b"
#LOAD MODEL
model = load_model(MODEL_URI)
encoder = load_encoder(run_id)
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
st.write('model used:')
st.write(model)
#format user input into dictionary (the select option default to zero)
dict ={'Pclass' : Pclass,
        'Sex' : Sex,
        'Age' : Age,
        'SibSp' : SibSp,
        'Parch' : Parch,
        'Fare'  : Fare,
        'Embarked' : Embarked}
#Convert dictionary into a dataframe
df = pd.DataFrame(dict, index=[0])
#encode and scale data
cat_col = ['Sex','Embarked']
df[cat_col]=encoder.transform(df[cat_col])
#make a prediction
if submit:
    outcome = model.predict(df).astype(np.float64)
    outcome = np.round(outcome*100,2)[0,0]
    st.subheader(f"Your probability of Surviving is : {outcome}%")
    if outcome > 80:
        st.success("WHAT ARE YOU DOING HERE?? GET TO THE EMMERGENCY RAFT RIGHT AWAY!")
    elif outcome > 30:
        st.warning("Best have luck on you, you might survive")
    elif outcome >= 0:
        st.error("Cry and Pray, you're going down with this ship.")
    else:
        st.error("somethig went wrong")