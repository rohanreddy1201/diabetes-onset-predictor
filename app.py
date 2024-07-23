import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# Load the models
lr_model = joblib.load('logistic_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')

# Load the dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Function to get user input
def user_input_features():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 1)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 20)
    insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 79.0)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    diabetes_pedigree = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Make predictions
lr_prediction = lr_model.predict(input_df)
rf_prediction = rf_model.predict(input_df)

st.title('Diabetes Onset Predictor')
st.write(f'Logistic Regression Prediction: {"Diabetic" if lr_prediction[0] else "Non-Diabetic"}')
st.write(f'Random Forest Prediction: {"Diabetic" if rf_prediction[0] else "Non-Diabetic"}')

# Display various graphs
st.subheader('Data Overview')
st.write(df.describe().iloc[:, :5])

st.subheader('Correlation Heatmap')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, ax=ax)
st.pyplot(fig)

st.subheader('Distribution of Outcome')
fig, ax = plt.subplots()
sns.countplot(x='Outcome', data=df, ax=ax)
ax.set_xlabel('Outcome')
ax.set_ylabel('Count')
st.pyplot(fig)

# Split pairplots into two separate plots for better readability
st.subheader('Pairplot of First Half of Features')
pairplot_features_1 = df.columns[:4]
fig = sns.pairplot(df, vars=pairplot_features_1, hue='Outcome', markers=["o", "s"])
st.pyplot(fig)

st.subheader('Pairplot of Second Half of Features')
pairplot_features_2 = df.columns[4:8]
fig = sns.pairplot(df, vars=pairplot_features_2, hue='Outcome', markers=["o", "s"])
st.pyplot(fig)

# Performance Metrics
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.subheader('Model Performance Comparison')

# Logistic Regression Metrics
lr_probs = lr_model.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
lr_cm = confusion_matrix(y_test, lr_model.predict(X_test))
lr_cm_norm = lr_cm.astype('float') / lr_cm.sum(axis=1)[:, np.newaxis]

st.write('### Logistic Regression Performance')
st.write(f'Accuracy: {accuracy_score(y_test, lr_model.predict(X_test)):.2f}')
st.write(f'Precision: {precision_score(y_test, lr_model.predict(X_test)):.2f}')
st.write(f'Recall: {recall_score(y_test, lr_model.predict(X_test)):.2f}')
st.write(f'F1 Score: {f1_score(y_test, lr_model.predict(X_test)):.2f}')
st.write(f'AUC: {lr_auc:.2f}')

fig, ax = plt.subplots()
ax.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression (AUC = %0.2f)' % lr_auc)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots()
sns.heatmap(lr_cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix - Logistic Regression')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# Random Forest Metrics
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
rf_cm = confusion_matrix(y_test, rf_model.predict(X_test))
rf_cm_norm = rf_cm.astype('float') / rf_cm.sum(axis=1)[:, np.newaxis]

st.write('### Random Forest Performance')
st.write(f'Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)):.2f}')
st.write(f'Precision: {precision_score(y_test, rf_model.predict(X_test)):.2f}')
st.write(f'Recall: {recall_score(y_test, rf_model.predict(X_test)):.2f}')
st.write(f'F1 Score: {f1_score(y_test, rf_model.predict(X_test)):.2f}')
st.write(f'AUC: {rf_auc:.2f}')

fig, ax = plt.subplots()
ax.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUC = %0.2f)' % rf_auc)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots()
sns.heatmap(rf_cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix - Random Forest')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)
