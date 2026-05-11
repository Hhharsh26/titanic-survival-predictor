import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/titanic_model.pkl")

# Sample passenger data
sample_passenger = pd.DataFrame([{
    'Pclass': 3,
    'Sex': 'male',
    'Age': 22,
    'SibSp': 1,
    'Parch': 0,
    'Fare': 7.25,
    'Embarked': 'S'
}])

# Predict
prediction = model.predict(sample_passenger)

if prediction[0] == 1:
    print("Passenger likely survived")
else:
    print("Passenger likely did not survive")