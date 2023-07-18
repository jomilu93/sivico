import pandas as pd
import os

from sivico.senator_matcher.matchers.tfidf_matcher.preprocessing import preprocess_text
from sivico.senator_matcher.matchers.tfidf_matcher.vectorization import fit_vectorizer, save_vectorizer_and_matrix, load_vectorizer_and_matrix

from sivico.senator_matcher.matchers.beto_matcher.embedding import generate_embeddings, save_embeddings

def preprocess() -> None:
    print("Preprocessing...")

    project_directory = os.getcwd()
    project_data_path = os.path.join(project_directory, 'data')

    df = pd.read_csv(os.path.join(project_data_path, 'senators_data_summarized_es.csv'))
    df['preprocessed_summary'] = df['initiative_summary_es'].apply(preprocess_text)
    df.to_csv(os.path.join(project_data_path, 'senators_data_summarized_es_preprocessed.csv'), index=False)

def vectorize_tfidf() -> None:
    print("Vectorizing...")

    project_directory = os.getcwd()
    project_model_path = os.path.join(project_directory, 'tfidf_model')

    matrix_filepath = os.path.join(project_model_path, 'tfidf_matrix_es.pkl')
    vectorizer_filepath = os.path.join(project_model_path, 'fitted_vectorizer_es.pkl')

    df = pd.read_csv(os.path.join(project_directory, 'data', 'senators_data_summarized_es_preprocessed.csv'))

    tfidf_matrix, vectorizer = fit_vectorizer(df, 'preprocessed_summary')
    save_vectorizer_and_matrix(tfidf_matrix, vectorizer, matrix_filepath, vectorizer_filepath)

def beto_embeddings() -> None:
    print("Generating Beto embeddings...")

    project_directory = os.getcwd()
    project_data_path = os.path.join(project_directory, 'data')

    project_model_path = os.path.join(project_directory, 'beto_embeddings')
    embeddings_filepath = os.path.join(project_model_path, 'embeddings_es.pkl')

    df = pd.read_csv(os.path.join(project_data_path, 'senators_data_summarized_es.csv'))
    embeddings = [generate_embeddings(text) for text in df['initiative_summary_es']]

    save_embeddings(embeddings, embeddings_filepath)