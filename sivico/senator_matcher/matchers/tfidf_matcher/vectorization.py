import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from google.cloud import storage
from sivico.params import *

matrix_filepath_storage = "tfidf_model/tfidf_matrix_es.pkl"
vectorizer_filepath_storage = "tfidf_model/fitted_vectorizer_es.pkl"

def fit_vectorizer(df, column):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df[column])
    return X, vectorizer

def save_vectorizer_and_matrix(X, vectorizer, matrix_filepath, vectorizer_filepath):
    with open(matrix_filepath, 'wb') as f:
        pickle.dump(X, f)

    with open(vectorizer_filepath, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_vectorizer_and_matrix(matrix_filepath, vectorizer_filepath):
    with open(matrix_filepath, 'rb') as f:
        matrix = pickle.load(f)

    with open(vectorizer_filepath, 'rb') as f:
        vectorizer = pickle.load(f)

    return vectorizer, matrix

def save_vectorizer_and_matrix_gc_storage(matrix_p, vectorizer_p):

    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(matrix_filepath_storage)
    with blob.open('wb') as f:
        pickle.dump(matrix_p, f)

    blob_v = bucket.blob(vectorizer_filepath_storage)
    with blob_v.open('wb') as f:
        pickle.dump(vectorizer_p, f)


def load_vectorizer_and_matrix_gc_storage():
    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(matrix_filepath_storage)
    with blob.open('rb') as f:
        matrix = pickle.load(f)

    blob_v = bucket.blob(vectorizer_filepath_storage)
    with blob_v.open('rb') as f:
        vectorizer = pickle.load(f)

    return vectorizer, matrix
