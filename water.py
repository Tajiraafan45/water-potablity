# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv("water_potability.csv")
    return data

data = load_data()

# Fill missing values with the median
data['ph'].fillna(data['ph'].median(), inplace=True)
data['Sulfate'].fillna(data['Sulfate'].median(), inplace=True)
data['Trihalomethanes'].fillna(data['Trihalomethanes'].median(), inplace=True)

# Preprocessing - Prepare data for training
X = data.drop('Potability', axis=1)
y = data['Potability']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Sidebar for user input
st.sidebar.title("Water Potability Predictor")

# Select classifier
classifier = st.sidebar.selectbox("Select Classifier", ("Decision Tree", "Random Forest", "SVM", "KNN", "XGBoost"))

# Train model
def train_model(classifier):
    if classifier == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif classifier == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif classifier == "SVM":
        model = SVC(probability=True, random_state=42)
    elif classifier == "KNN":
        model = KNeighborsClassifier()
    elif classifier == "XGBoost":
        model = XGBClassifier(random_state=42)
    
    model.fit(X_train, y_train)
    return model

model = train_model(classifier)

# User input for prediction
st.header("Input Water Test Parameters")

# Create input fields for each feature
ph = st.number_input("pH level", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness", min_value=0.0, max_value=500.0, value=200.0)
solids = st.number_input("Solids (ppm)", min_value=0.0, max_value=50000.0, value=10000.0)
chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, max_value=10.0, value=5.0)
sulfate = st.number_input("Sulfate (ppm)", min_value=0.0, max_value=500.0, value=250.0)
conductivity = st.number_input("Conductivity (µS/cm)", min_value=0.0, max_value=2000.0, value=800.0)
organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, max_value=30.0, value=10.0)
trihalomethanes = st.number_input("Trihalomethanes (ppm)", min_value=0.0, max_value=150.0, value=50.0)
turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=10.0, value=3.0)

# Store the input values into a dataframe
input_data = pd.DataFrame({
    'ph': [ph],
    'Hardness': [hardness],
    'Solids': [solids],
    'Chloramines': [chloramines],
    'Sulfate': [sulfate],
    'Conductivity': [conductivity],
    'Organic_carbon': [organic_carbon],
    'Trihalomethanes': [trihalomethanes],
    'Turbidity': [turbidity]
})

# Scale the input data
input_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict Potability"):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.write("The water is **potable**.")
    else:
        st.write("The water is **not potable**.")
