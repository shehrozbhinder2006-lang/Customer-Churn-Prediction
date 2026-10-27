
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = pd.read_csv(r"C:\Users\ITW\Downloads\telco_churn.csv")

# 2. Clean column names
data.columns = data.columns.str.strip()

# Remove customer ID if present
if "customerID" in data.columns:
    data = data.drop(columns=["customerID"])

# 3. Check target column
if "Churn" not in data.columns:
    raise ValueError(
        "CSV mein 'Churn' column nahi mila. "
        f"Available columns: {data.columns.tolist()}"
    )

# Remove rows with missing target
data = data.dropna(subset=["Churn"]).copy()

# Convert Yes/No to 1/0
y = data["Churn"].astype(str).str.strip().map({
    "Yes": 1,
    "No": 0
})

# Keep only valid target rows
valid = y.notna()
data = data.loc[valid].copy()
y = y.loc[valid].astype(int)

# Input features
X = data.drop(columns=["Churn"])

# Convert numeric-looking columns to numbers
for col in X.columns:
    if X[col].dtype == "object":
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().sum() > 0 and converted.notna().sum() >= 0.8 * X[col].notna().sum():
            X[col] = converted

# Identify numeric and categorical columns
num_features = X.select_dtypes(include="number").columns.tolist()
cat_features = X.select_dtypes(exclude="number").columns.tolist()

# Numeric preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Create model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model in the format required by app.py
saved_data = {
    "model": model,
    "features": X.columns.tolist()
}

with open("churn_model.pkl", "wb") as file:
    pickle.dump(saved_data, file)

print("\nchurn_model.pkl created successfully!")
