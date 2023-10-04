import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open("D:/deployment of ml model/trained_model.sav" , 'rb'))

#creating a function for prediction

def diabetes_pred(input_data):

    #changing the input_data into numpy array
    input_data_as_np_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instances
    input_data_reshaped = input_data_as_np_array.reshape(1 , -1)

    #standardize the input data
    # std_data = scalar.transform(input_data_reshaped)
    # print(std_data)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0] == 0):
        return 'The person is not diabetic'

    else:
        return 'the person is diabetic'
    
    
def main():
    #giving a title
    st.title('Diabetes prediction by subhabrata')
    
    #getting the input data from the user
    Pregnancies = st.number_input('number of pregnencies value')
    Glucose = st.number_input('number of glucose value')
    BloodPressure= st.number_input('number of BloodPressure value')
    SkinThickness  = st.number_input('number of SkinThickness value')
    BMI = st.number_input('number of BMI value')
    DiabetesPedigreeFunction = st.number_input('number of DiabetesPedigreeFunction value')
    Insulin = st.number_input('Insulin level')
    Age = st.number_input('age of a person')
    
    
    #code for prediction
    diagnosis = ''
    
    #creating a buuton for prediction
    if st.button('Diabetes test result'):
        diagnosis = diabetes_pred([Pregnancies,	Glucose,	BloodPressure,	SkinThickness,	Insulin,	BMI,	DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()

