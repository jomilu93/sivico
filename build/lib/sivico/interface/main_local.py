import pandas as pd

from senator_matcher.tfidf_matcher.preprocessing import preprocess_text

def preprocess() -> None:
    print("Preprocessing...")

    df = pd.read_csv('data/senators_data.csv').head(1)
    df['preprocessed_summary'] = df['initiatives_summary_dummy'].apply(preprocess_text)
    df.to_csv('data/senators_processed_data.csv', index=False)
