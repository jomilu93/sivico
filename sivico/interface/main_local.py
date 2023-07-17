import pandas as pd
import os

from sivico.senator_matcher.matchers.tfidf_matcher.preprocessing import preprocess_text

def preprocess() -> None:
    print("Preprocessing...")

    project_directory = os.getcwd()
    project_data_path = os.path.join(project_directory, 'data')

    df = pd.read_csv(os.path.join(project_data_path, 'senators_data_summarized_en.csv')).head(1)
    df['preprocessed_summary'] = df['initiative_summary_en'].apply(preprocess_text)
    df.to_csv(os.path.join(project_data_path, 'senators_data_summarized_en_preprocessed.csv'), index=False)
