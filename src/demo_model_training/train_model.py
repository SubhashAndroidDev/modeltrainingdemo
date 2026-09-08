import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# -----------------------------------------
# 1. Load historical data
# -----------------------------------------

df = pd.read_csv("data/loan_data.csv")

print("Dataset shape:", df.shape)


# -----------------------------------------
# 2. Separate features and target
# -----------------------------------------

X = df.drop("loan_approved", axis=1)

y = df["loan_approved"]


# -----------------------------------------
# 3. Identify columns
# -----------------------------------------

categorical_columns = [
    "employment_type",
    "city_tier",
]

numeric_columns = [
    "age",
    "monthly_income",
    "employment_years",
    "credit_score",
    "existing_emi",
    "loan_amount",
    "loan_term_months",
]


# -----------------------------------------
# 4. Preprocessing
# -----------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns,
        )
    ],
    remainder="passthrough",
)


# -----------------------------------------
# 5. Create ML model
# -----------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
)


# -----------------------------------------
# 6. Create pipeline
# -----------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model),
    ]
)


# -----------------------------------------
# 7. Train/Test split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# -----------------------------------------
# 8. Train
# -----------------------------------------

pipeline.fit(
    X_train,
    y_train,
)


print("\nModel training completed!")


# -----------------------------------------
# 9. Prediction
# -----------------------------------------

y_pred = pipeline.predict(X_test)


# -----------------------------------------
# 10. Accuracy
# -----------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred,
)

print("\nAccuracy:")
print(f"{accuracy:.2%}")


# -----------------------------------------
# 11. Detailed evaluation
# -----------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred,
    )
)


# -----------------------------------------
# 12. Save complete pipeline
# -----------------------------------------

joblib.dump(
    pipeline,
    "models/loan_model.pkl",
)


print(
    "\nModel saved to models/loan_model.pkl"
)