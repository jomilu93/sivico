import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# $WIPE_BEGIN
# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy Deep Learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!
# This will prove very useful for the Demo Day
# $WIPE_END

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2

@app.get("/senators")
def senators(user_input: str):
    df = pd.read_csv('data/senators_data_summarized_es.csv')

    matrix_filepath = 'tfidf_model/tfidf_matrix_es.pkl'
    vectorizer_filepath = 'tfidf_model/fitted_vectorizer_es.pkl'

    vectorizer, matrix = load_vectorizer_and_matrix(matrix_filepath, vectorizer_filepath)

    return match_senators(user_input, df, vectorizer, matrix).to_dict(orient='records')
