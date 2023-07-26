import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from sivico.params import *
from sivico.text_input_and_summarization.data import get_data_from_bq, load_data_to_bq
from sivico.senator_matcher.matchers.tfidf_matcher.preprocessing import preprocess_text
from sivico.senator_matcher.matchers.tfidf_matcher.vectorization import fit_vectorizer, save_vectorizer_and_matrix, load_vectorizer_and_matrix

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

def vectorize_tfidf() -> None:
    print("Vectorizing...")

    project_directory = os.getcwd()
    project_model_path = os.path.join(project_directory, 'tfidf_model')

    matrix_filepath = os.path.join(project_model_path, 'tfidf_matrix_es.pkl')
    vectorizer_filepath = os.path.join(project_model_path, 'fitted_vectorizer_es.pkl')

    df = get_data_from_bq('processed_senators')

    tfidf_matrix, vectorizer = fit_vectorizer(df, 'tfidf_preprocessed_summary')
    save_vectorizer_and_matrix(tfidf_matrix, vectorizer, matrix_filepath, vectorizer_filepath)

def beto_preprocess() -> None:
    print('...Preprocessing')

    df = get_data_from_bq('summarized_senators')

    df['beto_preprocessed_summary'] = df['BETO_summary'].apply(preprocess_text_for_beto)
    df.dropna(subset=['beto_preprocessed_summary'], inplace=True)

    load_data_to_bq(
        df,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_senators',
        truncate=True
    )
