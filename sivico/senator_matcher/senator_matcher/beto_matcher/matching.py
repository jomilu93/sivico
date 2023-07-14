from senator_matcher.beto_matcher.embedding import generate_embeddings

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def match_senators(user_input, senators_embeddings):
    # Generate an embedding for the user's input
    user_embedding = generate_embeddings(user_input)

    # Initialize an empty array to store the similarity scores
    scores = np.zeros(len(senators_embeddings))

    # Calculate the cosine similarity between the user's input and each senator's profile
    for i, senator_embedding in enumerate(senators_embeddings):
        scores[i] = cosine_similarity(user_embedding.detach().numpy().reshape(1, -1), senator_embedding.detach().numpy().reshape(1, -1))

    return scores

def get_top_senators(scores, senators_df, N=5):
    # Get the indices of the senators sorted by their scores
    sorted_indices = np.argsort(scores)[::-1]

    # Get the top N senators
    top_senators = senators_df.iloc[sorted_indices[:N]].copy()

    # Add similarity score to the dataframe
    top_senators['similarity_score'] = scores[sorted_indices[:N]]

    return top_senators[['Apellidos', 'Nombre', 'Fraccion', 'similarity_score']]