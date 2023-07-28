from sklearn.metrics.pairwise import cosine_similarity
from .preprocessing import preprocess_text

def match_senators(user_input, df, vectorizer, matrix, n=5):
    # Preprocess the user's input
    user_input_preprocessed = preprocess_text(user_input)

    # Transform the preprocessed user input into a TF-IDF vector
    user_vector = vectorizer.transform([user_input_preprocessed])

    # Calculate the cosine similarity between the user's vector and each senator's vector
    similarity_scores = cosine_similarity(user_vector, matrix)

    # Add the similarity scores to the dataframe
    df['similarity_score'] = similarity_scores[0]

    # Sort the dataframe by similarity score
    df_sorted = df.sort_values('similarity_score', ascending=False)

    # Return the sorted dataframe
    return df_sorted[['senator_id', 'senadores', 'Fraccion', 'Estado', 'correo', 'url_sitio', 
                    'telefono', 'attendance_score', 'similarity_score', 'Salud_initiative_list',
                    'Estudios_Legislativos_initiative_list', 'Educación_initiative_list', 
                    'Para_la_Igualdad_de_Género_initiative_list', 'Defensa_Nacional_initiative_list',
                    'Gobernación_initiative_list','Seguridad_Social_initiative_list',
                    'Anticorrupción__Transparencia_y_Participación_Ciudadana_initiative_list',
                    'Desarrollo_Urbano__Ordenamiento_Territorial_y_Vivienda_initiative_list',
                    'Justicia_initiative_list','Derechos_de_la_Niñez_y_de_la_Adolescencia_initiative_list',
                    'Comunicaciones_y_Transportes_initiative_list','Economía_initiative_list',
                    'Medio_Ambiente__Recursos_Naturales_y_Cambio_Climático_initiative_list',
                    'Hacienda_y_Crédito_Público_initiative_list','Relaciones_Exteriores_initiative_list',
                    'Agricultura__Ganadería__Pesca_y_Desarrollo_Rural_initiative_list','Seguridad_Pública_initiative_list',
                    'Reglamentos_y_Prácticas_Parlamentarias_initiative_list','Derechos_Humanos_initiative_list',
                    'Asuntos_Fronterizos_y_Migratorios_initiative_list','Ciencia_y_Tecnología_initiative_list',
                    'Energía_initiative_list','Juventud_y_Deporte_initiative_list',
                    'Radio__Televisión_y_Cinematografía_initiative_list','Cultura_initiative_list','Minería_y_Desarrollo_Regional_initiative_list',
                    'Comunicaciones_y_Obras_Públicas_initiative_list','Turismo_initiative_list','Urgente_Resolución_initiative_list',
                    'Junta_de_Coordinación_Política_initiative_list','Federalismo_y_Desarrollo_Municipal_initiative_list',
                    'Marina_initiative_list','Asuntos_Indígenas_initiative_list','Zonas_Metropolitanas_y_Movilidad_initiative_list',
                    'Puntos_Constitucionales_initiative_list','Cámara_de_Diputados_initiative_list','Defensa_de_los_Consumidores_initiative_list',
                    'Mesa_Directiva_initiative_list']].head(n)
