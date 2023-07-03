import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBClassifier
from xgboost import plot_importance  

# Selection model
from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV,StratifiedKFold 

# Metrics
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score, confusion_matrix, classification_report

# Visualization
#import matplotlib.pyplot as plt

import pickle
grid_search = pickle.load(open('grid_search.pkl', 'rb'))

import streamlit as st

#grid_search = GridSearchCV(model, param_grid, scoring="accuracy", n_jobs = 10, cv=kfold) 


df=pd.read_excel('Bank_Personal_Loan_Modelling.xlsx',"Data")
x = df.drop(['Personal Loan'], axis = 1)

def preda(x):
    #df2 = df.loc[(df['id'] == x)].drop(columns = ['Personal Loan','id'], axis=1)
    model_best = grid_search.best_estimator_
    y_pred=model_best.predict(x)
    if(y_pred==0):
        return 0
    else:
        return 1

st.title('Credit Score Estimator')
age = st.slider('Age', 0, 130, 25)
st.write(age)

exp = st.slider('Number of years of professional experience', 0, 130, 25)
st.write(exp)

inc = st.slider('Anual Income', 0, 200, 25)
st.write(inc)

zip = st.text_input('Home Address ZIP code')
st.write(zip)

fam = st.selectbox(
    'Family Size',
    (1, 2, 3,4))
st.write(fam)

cavg = st.slider(
    'Average Spending on credit cards per month',
    0.0, 5.0)
st.write(cavg)

Edu = st.selectbox(
    'Education',
    ('Undergrade', 'Graduate', 'Advanced/Professional'))
st.write('You selected:', Edu)
if(Edu=='Undergrade'):
    Edu=1
elif(Edu=='Graduate'):
    Edu=2
elif(Edu=='Advanced/Professional'):
    Edu=3


mor = st.text_input('Value of house mortgage, if any (0 if none)')
st.write(mor)

Securities = st.checkbox('Do you have a Securities account with the bank?')
if Securities:
    s=1
else:
    s=0

cd = st.checkbox('Do you have a certificate of deposit (CD) account with the bank?')
if cd:
    c=1
else:
    c=0

online = st.checkbox('Do you use Internet banking facilities?')
if online:
    o=1
else:
    o=0

CreditCard = st.checkbox('Do you have a Credit Card issued by UniversalBank?')
if CreditCard:
    cc=1
else:
    cc=0


import time
if st.button('Check my Credit Score'):
    df1 = pd.DataFrame({"Age":[age],"Experience":[exp],"Income":[inc],"ZIP Code":[int(zip)],"Family":[fam],"CCAvg":[cavg],"Education":[Edu],"Mortgage":[int(mor)],"Securities Account":[s],"CD Account":[c],"Online":[o],"CreditCard":[cc]})
    st.dataframe(df1)

    my_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.001)
        my_bar.progress(percent_complete + 1)

    if ((preda(df1)) == 1):
        st.success("Credit Score: Good")
    else:
        st.error("Credit Score: Bad")