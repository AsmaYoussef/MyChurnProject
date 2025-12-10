# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Columns defined EXACTLY like in the notebook
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

categorical_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

def create_preprocessor():
    """
    Creates the preprocessing pipeline:
    - scales numeric features
    - one-hot encodes categorical features
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

def preprocess_data(df, preprocessor=None, fit=False):
    """
    Prepares raw data for ML:
    - Fixes TotalCharges
    - Fits preprocessor if needed
    """

    df = df.copy()

    # Fix TotalCharges issue
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)

    if preprocessor is None:
        preprocessor = create_preprocessor()

    if fit:
        return preprocessor.fit_transform(df), preprocessor
    else:
        return preprocessor.transform(df), preprocessor
