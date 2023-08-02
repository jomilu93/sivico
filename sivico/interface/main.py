import numpy as np
import pandas as pd
import pickle
from google.cloud import storage

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from sivico.params import *
from sivico.text_input_and_summarization.data import get_data_from_bq, get_senator_initiative_data, load_data_to_bq
from sivico.senator_matcher.matchers.tfidf_matcher.preprocessing import preprocess_text
from sivico.senator_matcher.matchers.tfidf_matcher.vectorization import fit_vectorizer, save_vectorizer_and_matrix_gc_storage
from sivico.senator_matcher.matchers.beto_matcher.preprocessing import preprocess_text_for_beto
from sivico.senator_matcher.matchers.beto_matcher.embedding import generate_embeddings, save_embeddings
from sivico.text_input_and_summarization.summarizer import summarize_beto

embeddings_filepath_storage = "beto_embeddings/embeddings_es.pkl"

# We want to run beto_preprocess first in ordert to have both
# preprocess fields (tfifd and BETO) in the same processed dataframe
def tfidf_preprocess() -> None:
    print("Preprocessing...")

    # we want to run the preprocessed in the already processed dataframe
    # to add the column tfidf_preprocessed_summary in the same dataframe
    # as beto_preprocessed_summary
    df = get_data_from_bq('processed_senators')
    #df.reset_index(drop=True, inplace=True)
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

def tfidf_batch_preprocess(batch_size=2) -> None:
    """
    Preprocesses the 'BETO_summary' column of the dataframe in batches
    and saves the result in 'processed_senators.csv'.
    """
    print("Preprocessing...")

    df = get_data_from_bq('processed_senators')

    # Dividing data into batches
    total_batches = len(df) // batch_size + (len(df) % batch_size != 0)

    # Iterate over batches and preprocess
    for i in range(total_batches):
        print(f'preprocessing batch {i + 1} of {total_batches}...')
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        df.loc[start_idx:end_idx, 'tfidf_preprocessed_summary'] = df.loc[start_idx:end_idx, 'BETO_summary'].apply(preprocess_text)

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
    print ("done fit vectorizer")
    save_vectorizer_and_matrix_gc_storage(tfidf_matrix, vectorizer)

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

    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(embeddings_filepath_storage)
    with blob.open('wb') as f:
        pickle.dump(all_embeddings, f)

    print("Embeddings saved successfully!")

def run_pipeline() -> None:
    get_senator_initiative_data()
    summarize_beto()
    beto_preprocess()
    beto_batch_embeddings()