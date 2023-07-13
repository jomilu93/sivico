from sklearn.metrics.pairwise import cosine_similarity
from .preprocessing import preprocess_text

def match_senators(user_input, df, vectorizer, X):
    # Preprocess the user's input
    user_input_preprocessed = preprocess_text(user_input)

    # Transform the preprocessed user input into a TF-IDF vector
    user_vector = vectorizer.transform([user_input_preprocessed])

    # Calculate the cosine similarity between the user's vector and each senator's vector
    similarity_scores = cosine_similarity(user_vector, X)

    # Add the similarity scores to the dataframe
    df['similarity_score'] = similarity_scores[0]

    # Sort the dataframe by similarity score
    df_sorted = df.sort_values('similarity_score', ascending=False)

    # Return the sorted dataframe
    return df_sorted[['Apellidos', 'Nombre', 'Fraccion', 'similarity_score']]
