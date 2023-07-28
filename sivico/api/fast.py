import pandas as pd
import pickle

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from sivico.senator_matcher.matchers.tfidf_matcher.matching import match_senators
from sivico.senator_matcher.matchers.tfidf_matcher.vectorization import load_vectorizer_and_matrix

from sivico.senator_matcher.matchers.beto_matcher.matching import match_senators as beto_match_senators
from sivico.senator_matcher.matchers.beto_matcher.matching import get_top_senators

from sivico.text_input_and_summarization.data import get_data_from_bq
from sivico.text_input_and_summarization.data import load_data_to_bq

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app_data = {}

# see https://fastapi.tiangolo.com/advanced/events/
@asynccontextmanager
async def lifespan(app: FastAPI):
    app_data['senators_df'] = get_data_from_bq("summarized_senators")

    matrix_filepath = 'tfidf_model/tfidf_matrix_es.pkl'
    vectorizer_filepath = 'tfidf_model/fitted_vectorizer_es.pkl'

    app_data['vectorizer'], app_data['matrix'] = load_vectorizer_and_matrix(matrix_filepath, vectorizer_filepath)

    # app_data['beto_senators_preprocessed'] = pd.read_csv('data/senators_data_updated_preprocessed.csv')

    with open('beto_embeddings/embeddings_es.pkl', 'rb') as f:
        app_data['beto_embeddings'] = pickle.load(f)

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/senators")
def senators(user_input: str):
    response = {}
    # response['tfidf'] = match_senators(
    #     user_input,
    #     app_data['senators_df'],
    #     app_data['vectorizer'],
    #     app_data['matrix']).to_dict(orient='records')

    response['beto'] = beto_senators(user_input).to_dict(orient='records')
    response['tfidf'] = beto_senators(user_input).to_dict(orient='records')
    return response

def beto_senators(user_input: str):
    scores = beto_match_senators(user_input, app_data['beto_embeddings'])
    return get_top_senators(scores, app_data['senators_df'] , N=5)
