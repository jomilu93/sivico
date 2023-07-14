from . import TOKENIZER, BETO_MODEL

import torch

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

    return embeddings