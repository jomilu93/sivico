import numpy as np
import pandas as pd
import pickle

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from sivico.params import *
from sivico.text_input_and_summarization.data import get_data_from_bq, load_data_to_bq
from sivico.senator_matcher.matchers.tfidf_matcher.preprocessing import preprocess_text
from sivico.senator_matcher.matchers.tfidf_matcher.vectorization import fit_vectorizer, save_vectorizer_and_matrix, load_vectorizer_and_matrix
from sivico.senator_matcher.matchers.beto_matcher.preprocessing import preprocess_text_for_beto

# We want to run beto_preprocess first in ordert to have both
# preprocess fields (tfifd and BETO) in the same processed dataframe
def tfidf_preprocess() -> None:
    print("Preprocessing...")

    # we want to run the preprocessed in the already processed dataframe
    # to add the column tfidf_preprocessed_summary in the same dataframe
    # as beto_preprocessed_summary
    df = get_data_from_bq('processed_senators')
    df['tfidf_preprocessed_summary'] = df['BETO_summary'].apply(preprocess_text)
    df.dropna(subset=['tfidf_preprocessed_summary'], inplace=True)
    df.reset_index(inplace=True)

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
    df.reset_index(inplace=True)

    load_data_to_bq(
        df,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_senators',
        truncate=True
    )

def beto_batch_embeddings(batch_size=2) -> None:
    print("Generating Beto embeddings in batches...")

    project_directory = os.getcwd()
    project_data_path = os.path.join(project_directory, 'data')

    df = get_data_from_bq('processed_senators')

    # Prepare the file to save embeddings incrementally
    project_model_path = os.path.join(project_directory, 'beto_embeddings')
    embeddings_filepath = os.path.join(project_model_path, 'embeddings_es.pkl')

    # Clear the existing embeddings file or create an empty one
    with open(embeddings_filepath, 'wb') as f:
        pickle.dump([], f)

    # Split the dataframe into batches and generate embeddings
    # this is to take into account partial batches.
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)

    for batch_number, batch in df.groupby(np.arange(len(df)) // batch_size):
        print(f"Processing batch {batch_number + 1} of {num_batches}...")
        batch_embeddings = [generate_embeddings(text) for text in batch['beto_preprocessed_summary']]

        # Load current embeddings, append new ones, and save back
        with open(embeddings_filepath, 'rb') as f:
            all_embeddings = pickle.load(f)
        all_embeddings.extend(batch_embeddings)
        with open(embeddings_filepath, 'wb') as f:
            pickle.dump(all_embeddings, f)

    print("Embeddings saved successfully!")
