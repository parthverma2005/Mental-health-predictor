import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load dataset
df = pd.read_csv("data/Mental Health Dataset.csv")
df = df[['family_history', 'Growing_Stress', 'Changes_Habits',
         'Mood_Swings', 'Coping_Struggles', 'Work_Interest',
         'Social_Weakness', 'treatment', 'self_employed', 'Days_Indoors',
         'Mental_Health_History', 'mental_health_interview', 'care_options']].dropna()

# Convert Yes/No to binary
binary_cols = df.columns[df.isin(['Yes', 'No']).any()]
df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if x == 'Yes' else 0)

# Binarize Days_Indoors
df['Days_Indoors'] = pd.to_numeric(df['Days_Indoors'], errors='coerce').fillna(0)
df['Days_Indoors'] = df['Days_Indoors'].apply(lambda x: 1 if x > 45 else 0)

# One-hot encoding if needed
df = pd.get_dummies(df)

# Split data
X = df.drop('treatment', axis=1)
y = df['treatment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
grid = GridSearchCV(xgb_clf, params, cv=3, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

# Calibrate
calibrated = CalibratedClassifierCV(grid.best_estimator_, method='isotonic', cv=3)
calibrated.fit(X_train, y_train)

# Evaluate
y_pred = calibrated.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
with open("models/xgb_mental_health_model.pkl", "wb") as f:
    pickle.dump(calibrated, f)

print("✅ XGBoost model trained and saved!")
