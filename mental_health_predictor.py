import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("data/Mental Health Dataset.csv")

# Select relevant columns
df = df[['family_history', 'Growing_Stress', 'Changes_Habits',
         'Mood_Swings', 'Coping_Struggles', 'Work_Interest',
         'Social_Weakness', 'treatment', 'self_employed', 'Days_Indoors',
         'Mental_Health_History', 'mental_health_interview', 'care_options']]

# Drop rows with missing values
df.dropna(inplace=True)

# Convert binary Yes/No columns to 1/0
for col in df.columns:
    if df[col].nunique() == 2 and set(df[col].unique()) <= {'Yes', 'No'}:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

# Convert Days_Indoors to numeric, fill NaNs, and binarize based on 45
df['Days_Indoors'] = pd.to_numeric(df['Days_Indoors'], errors='coerce')
df['Days_Indoors'].fillna(0, inplace=True)
df['Days_Indoors'] = df['Days_Indoors'].apply(lambda x: 1 if x > 45 else 0)

# One-hot encode any remaining categorical columns
df = pd.get_dummies(df)

# Split into features and label
X = df.drop('treatment', axis=1)
y = df['treatment']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open("models/mental_health_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")
