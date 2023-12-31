import pandas as pd
import numpy as np
import os
import pickle

from sivico.senator_matcher.matchers.tfidf_matcher.preprocessing import preprocess_text
from sivico.senator_matcher.matchers.tfidf_matcher.vectorization import fit_vectorizer, save_vectorizer_and_matrix, load_vectorizer_and_matrix

from sivico.senator_matcher.matchers.beto_matcher.embedding import generate_embeddings, save_embeddings
from sivico.senator_matcher.matchers.beto_matcher.preprocessing import preprocess_text_for_beto

def tfidf_preprocess() -> None:
    print("Preprocessing...")

    project_directory = os.getcwd()
    project_data_path = os.path.join(project_directory, 'data')

    df = pd.read_csv(os.path.join(project_data_path, 'summarized_senators.csv'))
    df['tfidf_preprocessed_summary'] = df['BETO_summary'].apply(preprocess_text)
    df.to_csv(os.path.join(project_data_path, 'processed_senators.csv'), index=False)

def tfidf_batch_preprocess(batch_size=2) -> None:
    """
    Preprocesses the 'BETO_summary' column of the dataframe in batches
    and saves the result in 'processed_senators.csv'.
    """
    print("Preprocessing...")

    project_directory = os.getcwd()
    project_data_path = os.path.join(project_directory, 'data')
    df = pd.read_csv(os.path.join(project_data_path, 'processed_senators.csv'))

    # Dividing data into batches
    total_batches = len(df) // batch_size + (len(df) % batch_size != 0)

    # Iterate over batches and preprocess
    for i in range(total_batches):
        print(f'preprocessing batch {i + 1} of {total_batches}...')
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        df.loc[start_idx:end_idx, 'tfidf_preprocessed_summary'] = df.loc[start_idx:end_idx, 'BETO_summary'].apply(preprocess_text)

    df.to_csv(os.path.join(project_data_path, 'processed_senators.csv'), index=False)

def vectorize_tfidf() -> None:
    print("Vectorizing...")

    project_directory = os.getcwd()
    project_model_path = os.path.join(project_directory, 'tfidf_model')

    matrix_filepath = os.path.join(project_model_path, 'tfidf_matrix_es.pkl')
    vectorizer_filepath = os.path.join(project_model_path, 'fitted_vectorizer_es.pkl')

    df = pd.read_csv(os.path.join(project_directory, 'data', 'processed_senators.csv'))

    tfidf_matrix, vectorizer = fit_vectorizer(df, 'tfidf_preprocessed_summary')
    save_vectorizer_and_matrix(tfidf_matrix, vectorizer, matrix_filepath, vectorizer_filepath)

def beto_preprocess() -> None:
    print('...Preprocessing')

    project_directory = os.getcwd()
    project_data_path = os.path.join(project_directory, 'data')

    df = pd.read_csv(os.path.join(project_data_path, 'senators_data_updated.csv'), index_col='Unnamed: 0')

    df['beto_preprocessed_summary'] = df['BETO_summary'].apply(preprocess_text_for_beto)
    df.dropna(subset=['beto_preprocessed_summary'], inplace=True)
    df.to_csv(os.path.join(project_data_path, 'processed_senators.csv'), index=False)

def beto_batch_embeddings(batch_size=2) -> None:
    print("Generating Beto embeddings in batches...")

    project_directory = os.getcwd()
    project_data_path = os.path.join(project_directory, 'data')

    df = pd.read_csv(os.path.join(project_data_path, 'processed_senators.csv'))

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