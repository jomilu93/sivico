import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from sivico.senator_matcher.matchers.tfidf_matcher.matching import match_senators
from sivico.senator_matcher.matchers.tfidf_matcher.vectorization import load_vectorizer_and_matrix

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
    app_data['senators_df'] = pd.read_csv('data/senators_data_summarized_es.csv')

    matrix_filepath = 'tfidf_model/tfidf_matrix_es.pkl'
    vectorizer_filepath = 'tfidf_model/fitted_vectorizer_es.pkl'

    app_data['vectorizer'], app_data['matrix'] = load_vectorizer_and_matrix(matrix_filepath, vectorizer_filepath)

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/senators")
def senators(user_input: str):
    return match_senators(
        user_input,
        app_data['senators_df'],
        app_data['vectorizer'],
        app_data['matrix']).to_dict(orient='records')
