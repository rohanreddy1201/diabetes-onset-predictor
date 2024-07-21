### Diabetes Onset Predictor

## Introduction
The Diabetes Onset Predictor project aims to predict the likelihood of diabetes onset in individuals using machine learning models. The project is built with Python and leverages the Streamlit library for interactive user interfaces.

## Project Structure

app.py: Main application script for running the Streamlit app.
diabetes_predictor.py: Script containing data processing and prediction logic.
logistic_regression_model.pkl: Serialized logistic regression model.
random_forest_model.pkl: Serialized random forest model.
Diabetes Onset Predictor.pyproj: Visual Studio project file.
Diabetes Onset Predictor.sln: Visual Studio solution file.

### Installation
## Clone the repository and install the required dependencies:
```bash
git clone https://github.com/rohanreddy1201/diabetes-onset-predictor.git
cd diabetes-onset-predictor
pip install -r requirements.txt
```
## Usage
Run the Streamlit application:
```bash
streamlit run app.py
```
This will start the application, allowing users to input patient data for diabetes prediction.

## Models
The project uses logistic regression and random forest models trained on a dataset containing various diabetes risk factors.

## Results
Models are evaluated based on accuracy, precision, recall, and F1 score.

## Contributing
Fork the repository, create a feature branch, commit changes, push the branch, and open a pull request.
