import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Load Kaggle Dataset
df = pd.read_csv("student_data.csv")
df = df.dropna()

# Encode text columns
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Target
X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Final Tuned Model
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=2,
    min_samples_split=4,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, pred))

# Save
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "columns.pkl")

print("Final Tuned Model Saved Successfully")