import pandas as pd
import pickle
import os
import gdown

def load_model(model_path):
    file_id = "1hcr1TYRtARLmb9rQBtXTfK6q0zaZCnw4"  # Replace with your actual file ID
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def preprocess_input(family_history, growing_stress, changes_habits, mood_swings,
                     coping_struggles, work_interest, social_weakness, self_employed,
                     days_indoors, mental_health_history, care_options, model):

    days_indoors_binary = 1 if days_indoors > 45 else 0

    input_data = {
        'family_history': [family_history],
        'Growing_Stress': [growing_stress],
        'Changes_Habits': [changes_habits],
        'Mood_Swings': [mood_swings],
        'Coping_Struggles': [coping_struggles],
        'Work_Interest': [work_interest],
        'Social_Weakness': [social_weakness],
        'self_employed': [self_employed],
        'Days_Indoors': [days_indoors_binary],
        'Mental_Health_History': [mental_health_history],
        'care_options': [care_options]
    }

    df_input = pd.DataFrame(input_data)

    for col in df_input.columns:
        if df_input[col].nunique() == 2 and set(df_input[col].unique()) <= {'Yes', 'No'}:
            df_input[col] = df_input[col].map({'Yes': 1, 'No': 0})

    df_input = pd.get_dummies(df_input)

    model_columns = pd.DataFrame(columns=model.feature_names_in_)
    df_input = df_input.reindex(columns=model_columns.columns, fill_value=0)

    return df_input.iloc[0]
