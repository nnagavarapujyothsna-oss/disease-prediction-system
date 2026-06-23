import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def train_model():
    data = pd.read_csv("Training.csv")

    data.fillna(0, inplace=True)

    X = data.drop("prognosis", axis=1)
    y = data["prognosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X.columns.tolist()

# Load Model
model, symptoms = train_model()

st.title("🩺 Disease Prediction System")

selected_symptoms = st.multiselect(
    "Select Symptoms",
    symptoms
)

if st.button("Predict"):

    if len(selected_symptoms) == 0:
        st.error("Please select at least one symptom")
        st.stop()

    user_input = [0] * len(symptoms)

    for symptom in selected_symptoms:
        idx = symptoms.index(symptom)
        user_input[idx] = 1

    prediction = model.predict([user_input])

    st.success(f"Predicted Disease: {prediction[0]}")