import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
import pickle
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model at startup
try:
    model = pickle.load(open("model.sav", "rb"))
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

@app.route("/")
def load_page():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    try:
        input_data = get_input_data(request.form)
        processed_data = preprocess_data(input_data)
        prediction, probability = make_prediction(processed_data)
        result = format_result(prediction, probability)
        return render_template('home.html', **result, **request.form)
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return render_template('error.html', error=str(e)), 500

def get_input_data(form_data) -> Dict[str, str]:
    """Extract input data from form."""
    fields = [
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure'
    ]
    return {field: form_data.get(f'query{i+1}', '') for i, field in enumerate(fields)}

def preprocess_data(data: Dict[str, str]) -> pd.DataFrame:
    """Preprocess the input data."""
    df = pd.DataFrame([data])
    
    # Convert SeniorCitizen to int
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    
    # Convert MonthlyCharges and TotalCharges to float
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    
    # Create tenure groups
    labels = [f"{i} - {i + 11}" for i in range(1, 72, 12)]
    df['tenure_group'] = pd.cut(df['tenure'].astype(float), range(1, 80, 12), right=False, labels=labels)
    df.drop(columns=['tenure'], inplace=True)
    
    # Create dummy variables
    categorical_columns = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod', 'tenure_group'
    ]
    
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    
    # Ensure all expected columns are present
    expected_columns = [
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges',
        'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes',
        'Dependents_No', 'Dependents_Yes', 'PhoneService_No', 'PhoneService_Yes',
        'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
        'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
        'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
        'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'PaperlessBilling_No', 'PaperlessBilling_Yes',
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
        'tenure_group_1 - 12', 'tenure_group_13 - 24', 'tenure_group_25 - 36',
        'tenure_group_37 - 48', 'tenure_group_49 - 60', 'tenure_group_61 - 72'
    ]
    
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    return df_encoded[expected_columns]

def make_prediction(processed_df: pd.DataFrame) -> tuple[int, float]:
    """Make a prediction using the loaded model."""
    prediction = model.predict(processed_df)[0]
    probability = model.predict_proba(processed_df)[0][1]
    return prediction, probability

def format_result(prediction: int, probability: float) -> dict:
    """Format the prediction result."""
    if prediction == 1:
        output1 = "This customer is likely to churn."
    else:
        output1 = "This customer is likely to continue."
    output2 = f"Confidence: {probability * 100:.2f}%"
    return {'output1': output1, 'output2': output2}

if __name__ == "__main__":
    app.run(debug=True)