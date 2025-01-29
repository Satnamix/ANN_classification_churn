import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


model= tf.keras.models.load_model('annclassification_practice/model.h5')

with open('annclassification_practice/label_encoder_gender.pkl','rb') as f:
    label_encoder_gender=pickle.load(f)
    
with open('annclassification_practice/onehot_encoder_geo.pkl','rb') as f:
    onehot_encoder_geo=pickle.load(f)

with open('annclassification_practice/scaler.pkl','rb') as f:
    scaler=pickle.load(f)

st.title("Customer Churn Prediction")

credit_score= st.number_input('Credit Score')
geography= st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender= st.selectbox('Gender',label_encoder_gender.classes_)
age= st.slider('Age',18,92)
tenure= st.slider('Tenure',0,10)
balance= st.number_input('Balance')
num_of_products= st.slider('Number of Products',1,4)
has_cr_card= st.selectbox('Has Credit Card',[0,1])
is_active_member= st.selectbox('Is Active Member',[0,1])
estimated_salary= st.number_input('Estimated Salary')

input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,
}


input_df=pd.DataFrame([input_data])
encode_geo=onehot_encoder_geo.transform([[input_data["Geography"]]]).toarray()
encode_geo_df=pd.DataFrame(encode_geo,columns=onehot_encoder_geo.get_feature_names_out())

input_df['Gender']=label_encoder_gender.transform(input_df['Gender'])

input_df=pd.concat([input_df.drop(['Geography'],axis=1),encode_geo_df],axis=1)

input_scaled=scaler.transform(input_df)

st.write(f'Churn Probability  : {model.predict(input_scaled)[0][0]*100:.2f} %')

if model.predict(input_scaled)[0][0]>0.5:   
    st.write("Custumor is unlikely to churn")
else:
    st.write("Custumor is likely to churn")
