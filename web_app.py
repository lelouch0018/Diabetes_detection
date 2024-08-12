# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:29:20 2024

@author: rcred
"""

import numpy as np
import pickle
import streamlit as st

#loading saved model
loaded_model=pickle.load(open("D:/ML rekts/svm_model.sav",'rb'))

#creating function for prediction
def diabetes_prediction(input_data):
    # making a predicitve system
    # gotta change the input data to numpy array
    in_data_np_arrary=np.asarray(input_data)
    # reshape array as we want prediciton for one instance
    data_reshaped=in_data_np_arrary.reshape(1,-1)
    # stdize input data
    #std_data=scaler.transform(data_reshaped)
    #print(std_data)
    prediction=loaded_model.predict(data_reshaped)
    print(prediction)
    if(prediction[0] == 0):
        return "Great! You are not diabetic."
    else:
        return "Ouch! You are diabetic, please give attention to your lifestyle.."


def main():
    st.title('Ladies,come check if you are diabetic.')
    #getting input data
    Pregnancies=st.text_input('Number of pregnancies')
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input("BloodPressure")
    SkinThickness=st.text_input('SkinThickness')
    Insulin=st.text_input('Insulin level')
    BMI=st.text_input('Body Mass Index')
    DiabetesPedigreeFunction=st.text_input('Pedigree function value')
    Age=st.text_input('Age of the person')
    
    #code for prediction
    diagnosis=''
    
    #button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    
    st.success(diagnosis)


if __name__=='__main__':
    main()                               