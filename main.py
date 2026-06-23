import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Train Model Function
def train_model():
    data = pd.read_csv("Training.csv")

    # Replace missing values
    data.fillna(0, inplace=True)

    # Features and target
    X = data.drop("prognosis", axis=1)
    y = data["prognosis"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model


# User Input Function
def get_user_input():
    print("\nEnter Symptoms (1 = Yes, 0 = No)\n")

    fever = int(input("Fever (1/0): "))
    cough = int(input("Cough (1/0): "))
    headache = int(input("Headache (1/0): "))
    fatigue = int(input("Fatigue (1/0): "))
    vomiting = int(input("Vomiting (1/0): "))
    body_pain = int(input("Body Pain (1/0): "))

    return [[fever, cough, headache, fatigue, vomiting, body_pain]]


# Main Program
model = train_model()

user_data = get_user_input()

prediction = model.predict(user_data)

print("\nPredicted Disease:", prediction[0])

