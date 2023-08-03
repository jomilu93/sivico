import pandas as pd

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import simplejson as json
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from sivico.senator_matcher.matchers.beto_matcher.matching import match_senators as beto_match_senators
from sivico.senator_matcher.matchers.beto_matcher.matching import get_top_senators
from sivico.senator_matcher.matchers.beto_matcher.embedding import load_embeddings_beto_gc_storage

from sivico.text_input_and_summarization.data import get_data_from_bq

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
    app_data['senators_df'] = get_data_from_bq("processed_senators")
    app_data['beto_embeddings'] = load_embeddings_beto_gc_storage()

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/senators")
def senators(user_input: str):
    data = beto_senators(user_input).to_dict(orient='records')
    clean_data = json.dumps(data, ignore_nan=True)

    return json.loads(clean_data)

def beto_senators(user_input: str):
    scores = beto_match_senators(user_input, app_data['beto_embeddings'])
    return get_top_senators(scores, app_data['senators_df'] , N=5)
