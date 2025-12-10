import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import preprocess_data

print("Training started...")

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop("customerID", axis=1, inplace=True)

# Encode target
le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"])

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train_processed, preprocessor = preprocess_data(X_train, fit=True)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train_processed, y_train)

joblib.dump(model, "churn_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

print("Model & preprocessor saved.")
