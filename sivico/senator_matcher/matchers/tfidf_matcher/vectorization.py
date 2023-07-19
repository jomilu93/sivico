import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

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
