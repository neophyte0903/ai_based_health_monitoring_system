import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')

# Feature columns and label
feature_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
                   'Gender', 'Blood Pressure', 'Cholesterol Level', 'Age']
label_column = 'Disease'

# Split data into features and target
X = df[feature_columns]
y = df[label_column]

# Encode categorical columns
categorical_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
                       'Gender', 'Blood Pressure', 'Cholesterol Level']

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Standardize numerical column (Age)
scaler = StandardScaler()
X['Age'] = scaler.fit_transform(X[['Age']])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict function
def predict_disease(user_input):
    input_data = {}

    # Transform categorical features
    for col in categorical_columns:
        if user_input[col] in label_encoders[col].classes_:
            input_data[col] = label_encoders[col].transform([user_input[col]])[0]
        else:
            input_data[col] = 0

    # Standardize Age
    input_data['Age'] = scaler.transform([[user_input['Age']]])[0][0]

    # Prepare input DataFrame
    feature_order = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
                     'Gender', 'Blood Pressure', 'Cholesterol Level', 'Age']
    input_df = pd.DataFrame([input_data])[feature_order]

    # Make prediction
    prediction = model.predict(input_df)
    return prediction[0]  # Predicted Disease

# Streamlit UI
st.title("AI Health Monitoring System")
st.header("Enter your symptoms")

# Create input fields in Streamlit
fever = st.selectbox('Fever (Yes/No)', ['Yes', 'No'])
cough = st.selectbox('Cough (Yes/No)', ['Yes', 'No'])
fatigue = st.selectbox('Fatigue (Yes/No)', ['Yes', 'No'])
breathing = st.selectbox('Difficulty Breathing (Yes/No)', ['Yes', 'No'])
age = st.number_input('Age', min_value=0, max_value=150, value=30)
gender = st.selectbox('Gender (Male/Female)', ['Male', 'Female'])
blood_pressure = st.selectbox('Blood Pressure (High/Normal/Low)', ['High', 'Normal', 'Low'])
cholesterol = st.selectbox('Cholesterol Level (Normal/High/Low)', ['Normal', 'High', 'Low'])

# Create a dictionary of inputs
user_input = {
    'Fever': fever,
    'Cough': cough,
    'Fatigue': fatigue,
    'Difficulty Breathing': breathing,
    'Age': age,
    'Gender': gender,
    'Blood Pressure': blood_pressure,
    'Cholesterol Level': cholesterol
}

# Button to trigger prediction
if st.button('Predict Disease'):
    prediction = predict_disease(user_input)
    st.write(f"Predicted Disease: {prediction}")

