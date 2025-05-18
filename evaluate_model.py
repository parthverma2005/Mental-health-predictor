import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load your full dataset
data = pd.read_csv("data/Mental Health Dataset.csv")

# Preprocess data just like in training
# Select relevant columns
data = data[['family_history', 'Growing_Stress', 'Changes_Habits',
             'Mood_Swings', 'Coping_Struggles', 'Work_Interest',
             'Social_Weakness', 'treatment', 'self_employed', 'Days_Indoors',
             'Mental_Health_History', 'mental_health_interview', 'care_options']]

# Drop missing values
data.dropna(inplace=True)

# Map Yes/No to 1/0
for col in data.columns:
    if data[col].nunique() == 2 and set(data[col].unique()) <= {'Yes', 'No'}:
        data[col] = data[col].map({'Yes': 1, 'No': 0})

# Process Days_Indoors
data['Days_Indoors'] = pd.to_numeric(data['Days_Indoors'], errors='coerce')
data['Days_Indoors'].fillna(0, inplace=True)
data['Days_Indoors'] = data['Days_Indoors'].apply(lambda x: 1 if x > 45 else 0)

# One-hot encode remaining categorical columns
data = pd.get_dummies(data)

# Split features and label
X = data.drop('treatment', axis=1)
y = data['treatment']

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load your saved model
with open("models/mental_health_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict on test set
y_pred = model.predict(X_test)

# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
