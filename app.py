import sqlite3
import streamlit as st
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline

conn = sqlite3.connect('user_data.db')
c = conn.cursor()

c.execute('''
          CREATE TABLE IF NOT EXISTS user_data
          (CreditScore INT, Geography TEXT, Gender TEXT, Age INT, 
           Tenure INT, Balance FLOAT, NumOfProducts INT, HasCrCard INT, 
           IsActiveMember INT, EstimatedSalary FLOAT, Prediction TEXT)
          ''')

st.title("Customer Churn Prediction")

CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, step=1, key="CreditScore")
Geography = st.selectbox("Geography", options=["France", "Germany", "Spain"], key="Geography")
Gender = st.selectbox("Gender", options=["Male", "Female"], key="Gender")
Age = st.slider("Age", min_value=18, max_value=92, key="Age")
Tenure = st.slider("Tenure", min_value=0, max_value=10, key="Tenure")
Balance = st.number_input("Balance", min_value=0.0, step=0.01, key="Balance")
NumOfProducts = st.slider("Number of Products", min_value=1, max_value=4, key="NumOfProducts")
HasCrCard = st.selectbox("Has Credit Card?", options=[0, 1], key="HasCrCard")
IsActiveMember = st.selectbox("Is Active Member?", options=[0, 1], key="IsActiveMember")
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, step=0.01, key="EstimatedSalary")


if st.button("Predict"):
    try:
        # Verileri CustomData sınıfına veriyoruz
        custom_data = CustomData(
            CreditScore=CreditScore,
            Geography=Geography,
            Gender=Gender,
            Age=Age,
            Tenure=Tenure,
            Balance=Balance,
            NumOfProducts=NumOfProducts,
            HasCrCard=HasCrCard,
            IsActiveMember=IsActiveMember,
            EstimatedSalary=EstimatedSalary,
        )

        features = custom_data.preprocess_features()

        predict_pipeline = PredictionPipeline()
        prediction = predict_pipeline.predict(features)

        if prediction == 1:
            st.success("The customer is likely to churn.")
        else:
            st.success("The customer is not likely to churn.")
        c.execute('''
            INSERT INTO user_data (CreditScore, Geography, Gender, Age, Tenure, Balance, 
                                   NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard,
              IsActiveMember, EstimatedSalary, prediction))

        conn.commit()
        st.success("User data saved to the database.")

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")