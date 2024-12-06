import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@st.cache_data
def load_and_preprocess_data():
    file_path = 'lifestyle.csv'
    lifestyle_data = pd.read_csv(file_path)
    
    # Define target column based on thresholds for "Healthy" or "Unhealthy"
    def classify_health(row):
        if (row['Stress Level'] > 7 or 
            row['Quality of Sleep'] < 5 or 
            row['BMI Category'] in ['Overweight', 'Obese'] or 
            row['Daily Steps'] < 5000):
            return "Unhealthy"
        else:
            return "Healthy"
        
    lifestyle_data['Health Status'] = lifestyle_data.apply(classify_health, axis=1)

    
    categorical_columns = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder', 'Health Status']
    label_encoders = {col: LabelEncoder() for col in categorical_columns}

    for col in categorical_columns:
        lifestyle_data[col] = label_encoders[col].fit_transform(lifestyle_data[col])

    return lifestyle_data, label_encoders


@st.cache_resource
def train_model(data):
    
    drop_columns = ['Health Status', 'Blood Pressure']  
    if 'Person ID' in data.columns:
        drop_columns.append('Person ID')
        
    X = data.drop(columns=drop_columns)
    y = data['Health Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_clf.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, rf_clf.predict(X_test))
    return rf_clf, accuracy, X.columns

# Generate health recommendations
def generate_recommendations(input_data):
    if input_data['Stress Level'] > 7:
        return "Consider stress management techniques like meditation, yoga, or counseling."
    elif input_data['Quality of Sleep'] < 5:
        return "Focus on improving sleep hygiene, such as consistent sleep schedules and a calming bedtime routine."
    elif input_data['Daily Steps'] < 5000:
        return "Increase daily physical activity to at least 7,000-10,000 steps."
    elif input_data['BMI Category'] in [2, 3]:  # Assuming 2=Overweight, 3=Obese
        return "Adopt a balanced diet and regular exercise to achieve a healthy BMI."
    else:
        return "Keep up the good work! Maintain your current healthy lifestyle."

# Streamlit App
st.title("Personalized Health Recommendation System")

# Load data and train model
data, encoders = load_and_preprocess_data()
model, accuracy, feature_columns = train_model(data)

st.sidebar.header("User Input Features")
# User inputs
user_input = {}
raw_data = pd.read_csv('lifestyle.csv')  # Load original raw dataset for ranges
for feature in feature_columns:
    if feature in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
        user_input[feature] = st.sidebar.selectbox(feature, encoders[feature].classes_)
    else:
        min_value = raw_data[feature].min()
        max_value = raw_data[feature].max()
        # Ensure the step size is consistent with the min and max types (int for BMI, Age, etc.)
        if np.issubdtype(raw_data[feature].dtype, np.int64):
            user_input[feature] = st.sidebar.slider(feature, min_value, max_value, int(raw_data[feature].mean()))
        else:
            user_input[feature] = st.sidebar.slider(feature, min_value, max_value, float(raw_data[feature].mean()))

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

# Apply label encoding to categorical features in user input
for feature in encoders:
    if feature in input_df.columns:
        input_df[feature] = encoders[feature].transform(input_df[feature])

# Check for numerical columns before scaling
numerical_columns = input_df.select_dtypes(include=['int64', 'float64']).columns
if not numerical_columns.empty:
    scaler = StandardScaler()
    input_df[numerical_columns] = scaler.fit_transform(input_df[numerical_columns])

# Prediction and recommendation
prediction = model.predict(input_df)[0]
health_status = 'Healthy' if prediction == 0 else 'Unhealthy'
recommendation = generate_recommendations(user_input)

st.write(f"### Predicted Health Status: {health_status}")
st.write("### Recommendation:")
st.write(recommendation)
