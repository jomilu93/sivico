from . import TOKENIZER, BETO_MODEL

import torch
import pickle
from google.cloud import storage

from sivico.params import *

def generate_embeddings(text, max_length=512):
    # Split the text into chunks to handle long summaries
    text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]

    # Initialize an empty tensor to store the embeddings
    embeddings = torch.zeros((len(text_chunks), BETO_MODEL.config.hidden_size))

    # Generate embeddings for each chunk
    for i, chunk in enumerate(text_chunks):
        inputs = TOKENIZER(chunk, return_tensors="pt", truncation=True, padding=True)
        outputs = BETO_MODEL(**inputs)
        embeddings[i] = outputs.last_hidden_state.mean(dim=1)

    # Average the embeddings of all the chunks
    embeddings = embeddings.mean(dim=0)

    # Returns the embedding of one summary
    return embeddings

def save_embeddings(embeddings, embeddings_filepath):
    with open(embeddings_filepath, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings_beto_gc_storage():
    embeddings_filepath_storage = "beto_embeddings/embeddings_es.pkl"

    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(embeddings_filepath_storage)
    with blob.open('rb') as f:
        embeddings = pickle.load(f)

    return embeddings