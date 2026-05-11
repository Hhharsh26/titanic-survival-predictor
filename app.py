import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/titanic_model.pkl")

st.title("Titanic Survival Predictor")
st.write("Predict whether a Titanic passenger would survive.")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses", 0, 10, 0)
parch = st.number_input("Number of Parents/Children", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

if st.button("Predict"):
    passenger_data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])

    prediction = model.predict(passenger_data)

    if prediction[0] == 1:
        st.success("Passenger likely survived")
    else:
        st.error("Passenger likely did not survive")