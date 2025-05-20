import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

df = pd.read_csv("data/Mental Health Dataset.csv")
df = df[['family_history', 'Growing_Stress', 'Changes_Habits',
         'Mood_Swings', 'Coping_Struggles', 'Work_Interest',
         'Social_Weakness', 'treatment', 'self_employed', 'Days_Indoors',
         'Mental_Health_History', 'mental_health_interview', 'care_options']]
df.dropna(inplace=True)

for col in df.columns:
    if df[col].nunique() == 2 and set(df[col].unique()) <= {'Yes', 'No'}:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

df['Days_Indoors'] = pd.to_numeric(df['Days_Indoors'], errors='coerce')
df['Days_Indoors'].fillna(0, inplace=True)
df['Days_Indoors'] = df['Days_Indoors'].apply(lambda x: 1 if x > 45 else 0)

df = pd.get_dummies(df)

X = df.drop('treatment', axis=1)
y = df['treatment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

param_grid = {
    'n_estimators': [200],
    'max_depth': [10],
    'min_samples_split': [10],
    'min_samples_leaf': [4],
    'bootstrap': [True]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

calibrated_model = CalibratedClassifierCV(best_rf, method='isotonic', cv=3)
calibrated_model.fit(X_train, y_train)

y_pred = calibrated_model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
with open("models/mental_health_model.pkl", "wb") as f:
    pickle.dump(calibrated_model, f)

print("✅ Calibrated model saved successfully!")
