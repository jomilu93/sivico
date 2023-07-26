import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from sivico.params import *
from sivico.text_input_and_summarization.data import get_data_from_bq, load_data_to_bq
from sivico.senator_matcher.matchers.tfidf_matcher.preprocessing import preprocess_text

def tfidf_preprocess() -> None:
    print("Preprocessing...")

    df = get_data_from_bq('summarized_senators')
    df['tfidf_preprocessed_summary'] = df['initiative_summary_es'].apply(preprocess_text)
    df.dropna(subset=['tfidf_preprocessed_summary'], inplace=True)

    load_data_to_bq(
        df,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_senators',
        truncate=True
    )

