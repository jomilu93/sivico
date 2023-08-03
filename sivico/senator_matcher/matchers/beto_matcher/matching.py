from .embedding import generate_embeddings

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

    return top_senators[['senator_id', 'senadores', 'Fraccion', 'Estado', 'correo', 'url_sitio',
                         'telefono', 'attendance_score', 'similarity_score', 'Salud_initiative_list',
                         'Estudios_Legislativos_initiative_list', 'Educacion_initiative_list',
                         'Para_la_Igualdad_de_Genero_initiative_list', 'Defensa_Nacional_initiative_list',
                         'Gobernacion_initiative_list','Seguridad_Social_initiative_list',
                         'Anticorrupcion__Transparencia_y_Participacion_Ciudadana_initiative_list',
                         'Desarrollo_Urbano__Ordenamiento_Territorial_y_Vivienda_initiative_list',
                         'Justicia_initiative_list','Derechos_de_la_Ninez_y_de_la_Adolescencia_initiative_list',
                         'Comunicaciones_y_Transportes_initiative_list','Economia_initiative_list',
                         'Medio_Ambiente__Recursos_Naturales_y_Cambio_Climatico_initiative_list',
                         'Hacienda_y_Credito_Publico_initiative_list','Relaciones_Exteriores_initiative_list',
                         'Agricultura__Ganaderia__Pesca_y_Desarrollo_Rural_initiative_list','Seguridad_Publica_initiative_list',
                         'Reglamentos_y_Practicas_Parlamentarias_initiative_list','Derechos_Humanos_initiative_list',
                         'Asuntos_Fronterizos_y_Migratorios_initiative_list','Ciencia_y_Tecnologia_initiative_list',
                         'Energia_initiative_list','Juventud_y_Deporte_initiative_list',
                         'Radio__Television_y_Cinematografia_initiative_list','Cultura_initiative_list','Mineria_y_Desarrollo_Regional_initiative_list',
                         'Comunicaciones_y_Obras_Publicas_initiative_list','Turismo_initiative_list','Urgente_Resolucion_initiative_list',
                         'Junta_de_Coordinacion_Politica_initiative_list','Federalismo_y_Desarrollo_Municipal_initiative_list',
                         'Marina_initiative_list','Asuntos_Indigenas_initiative_list','Zonas_Metropolitanas_y_Movilidad_initiative_list',
                         'Puntos_Constitucionales_initiative_list','Camara_de_Diputados_initiative_list','Defensa_de_los_Consumidores_initiative_list',
                         'Mesa_Directiva_initiative_list', 'beto_preprocessed_summary']]