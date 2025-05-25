import pandas as pd
import pickle
import os
import gdown

def load_model(model_path):
    file_id = "1aMwYa9gfv7aSbu6_LY4gO-5hlCT9kyGD"  # Update if needed  https://drive.google.com/file/d/1aMwYa9gfv7aSbu6_LY4gO-5hlCT9kyGD/view?usp=sharing
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def preprocess_input(
    family_history, growing_stress, changes_habits, mood_swings,
    coping_struggles, work_interest, social_weakness, self_employed,
    mental_health_history, care_options, model
):
    input_data = {
        'family_history': [family_history],
        'Growing_Stress': [growing_stress],
        'Changes_Habits': [changes_habits],
        'Mood_Swings': [mood_swings],
        'Coping_Struggles': [coping_struggles],
        'Work_Interest': [work_interest],
        'Social_Weakness': [social_weakness],
        'self_employed': [self_employed],
        'Mental_Health_History': [mental_health_history],
        'care_options': [care_options]
    }

    df_input = pd.DataFrame(input_data)

    # Convert Yes/No to 1/0
    yes_no_columns = df_input.columns[df_input.isin(['Yes', 'No']).any()]
    df_input[yes_no_columns] = df_input[yes_no_columns].replace({'Yes': 1, 'No': 0})

    # One-hot encoding if needed
    df_input = pd.get_dummies(df_input)

    # Align columns with model's features
    model_columns = pd.DataFrame(columns=model.feature_names_in_)
    df_input = df_input.reindex(columns=model_columns.columns, fill_value=0)

    return df_input.iloc[0]
