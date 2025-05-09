import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("data/Mental Health Dataset.csv")

# Select and clean necessary columns
df = df[['family_history', 'Growing_Stress', 'Changes_Habits',
         'Mood_Swings', 'Coping_Struggles', 'Work_Interest',
         'Social_Weakness', 'treatment', 'self_employed', 'Days_Indoors',
         'Mental_Health_History', 'mental_health_interview', 'care_options']]

# Drop rows with missing values
df.dropna(inplace=True)

# Convert Yes/No to binary (1/0) where applicable
for col in df.columns:
    if df[col].nunique() == 2 and set(df[col].unique()) <= {'Yes', 'No'}:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

# One-hot encode remaining categorical columns (like '31-60 days')
df = pd.get_dummies(df)

# Separate features and target
X = df.drop('treatment', axis=1)
y = df['treatment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
with open("models/mental_health_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")
