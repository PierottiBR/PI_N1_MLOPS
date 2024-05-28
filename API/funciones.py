import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
from sklearn.metrics.pairwise import cosine_similarity
import traceback

df_steam_games = pd.read_csv(os.path.join("datasets","steamgames.csv"))

df_user_reviews = pd.read_csv(os.path.join("datasets","userreviews.csv"))

df_user_items= pd.read_csv(os.path.join("datasets","useritems.csv"))

df_reviews_limpio = pd.read_csv(os.path.join("datasets","reviews_limpio.csv"))

#UserForGenre Funcion
def UserForGenre(genero):
    
    filtrado_por_genero = df_steam_games[df_steam_games['genres'].str.contains(genero, case=False, na=False)]

    lista_item_ids = filtrado_por_genero['item_id'].tolist()

    filtrado_por_ids = df_user_items[df_user_items['item_id'].isin(lista_item_ids)]

    suma_horas_por_usuario = filtrado_por_ids.groupby('user_id')['playtime_forever'].sum().reset_index()

    usuario_obtenido = suma_horas_por_usuario.max()['user_id']

    df_usuario = df_user_items[df_user_items['user_id']==usuario_obtenido]
    lista_item_ids = df_usuario['item_id'].tolist()

    lista_juegos = df_steam_games[df_steam_games['item_id'].isin(lista_item_ids)]
    
    dict_resultado = {}

    for indice, fila in lista_juegos.iterrows():
        año = fila['release_date'].split('-')[0]
        if año in dict_resultado.keys():
            dict_resultado[año] += int(df_usuario[df_usuario['item_id']==fila['item_id']]['playtime_forever'].sum())
        else:
            dict_resultado[año] = int(df_usuario[df_usuario['item_id']==fila['item_id']]['playtime_forever'].sum())

    return {f'Usuario con más horas jugadas para genero {genero}' : usuario_obtenido, 'Horas jugadas':dict_resultado}

#Datos para UserData Funcion
# Convertir la columna 'item_id' en df_user_reviews a tipo de datos 'float64'
df_user_reviews['item_id'] = df_user_reviews['item_id'].astype(float)

# Ahora, puedes fusionar los DataFrames sin generar un error de tipo
df_reviews_games2 = df_user_reviews.merge(df_steam_games[['item_id', 'price']])

#UserData Funcion
def userdata(user_id):
    # Filtro de datos para usuario especifico
    user_data = df_reviews_games2[df_reviews_games2['user_id'] == user_id]
    if user_data.empty:
        return f'No hay reviews para el usuario {user_id}'
    
    # Calcular el dinero gastado por el usuario
    spent_money = user_data['price'].sum()

    # Calcular el porcentaje de recomendacion
    recommendations = (user_data['recommend'] == True).count()
    recommendation_percentage = recommendations / len(user_data) * 100

    # Calcular el numero de items
    number_of_items = user_data['item_id'].nunique()

    # Creacion de diccionario para resultado
    results = {
        'Cantidad de dinero gastado': round(spent_money, 2),
        'Porcentaje de recomendacion': recommendation_percentage,
        'Numero de items': number_of_items
    }
    
    return results

#Developer Funcion
def developer(developer):
    
    def extraer_año(fecha):
        return fecha.split('-')[0]

    df_developer = df_steam_games[df_steam_games['developer'].str.lower() == developer.lower()]
    df_developer['year'] = df_developer['release_date'].apply(extraer_año)
    
    items_por_anio = df_developer.groupby('year')['item_id'].count()

    df_dev_free = df_developer[df_developer['price'] == 0]
 
    free_items = df_dev_free.groupby('year')['price'].count() 
 
    free_proportion = round((free_items / items_por_anio) * 100, 2)

    items_por_anio.name = 'Number of Items'
    free_proportion.name = 'Free Content'

    df1 = pd.merge(items_por_anio, free_proportion, on='year').reset_index()
    df1 = df1.fillna(0)

    df1['Free Content'] = df1['Free Content'].apply(lambda x: f'{x}%')

    return df1.to_json(orient='records')


#Dataframes para Funcion
df_games_reviews5 = pd.merge(df_steam_games, df_reviews_limpio, how='inner', left_on='item_id', right_on='item_id')

df_games_reviews5['release_date'] = pd.to_datetime(df_games_reviews5['release_date'], format='%Y-%m-%d', errors='coerce')

# Extraer el año y almacenarlo en una nueva columna llamada 'release_year'
df_games_reviews5['release_year'] = df_games_reviews5['release_date'].dt.year
df_games_reviews5['release_year'] = df_games_reviews5['release_year'].astype('Int64')

#Mejor desarrollador del anio Funcion
def best_developer_year(year):
    df = df_games_reviews5[df_games_reviews5['release_year']==int(year)]

    df = df[(df['recommend']==True) & (df['sentiment_analysis']==2)]
    
    df_final = df.groupby('developer').count().reset_index()
    df_final = df_final.sort_values('sentiment_analysis', ascending=False)
    
    lista_mejores = df_final.iloc[:3]['developer'].tolist()
    lista_resultado=[]
    for indice,dev in enumerate(lista_mejores):
        lista_resultado.append({f'puesto_{indice+1}':dev})
        
    return lista_resultado

def reviews_analysis(desarrolladora):
    df_filtrado = df_games_reviews5[df_games_reviews5['developer'].str.lower()==desarrolladora.lower()]
    
    negativo = len(df_filtrado[(df_filtrado['sentiment_analysis']==0)])
    positivo = len(df_filtrado[(df_filtrado['sentiment_analysis']==2)])
    
    return {desarrolladora : {'Negative':negativo, 'Positive':positivo}}

#Sistema de recomendacion

def recomendacion_juego(product_id: str):
    try:
        # Obtener el ID del juego
        target_game = df_steam_games[df_steam_games['item_name'].str.lower() == product_id.lower()]

        if target_game.empty:
            return {"message": "No se encontró el juego de referencia."}

        # Combina las etiquetas (tags) y géneros en una sola cadena de texto
        target_game_tags_and_genres = ' '.join(target_game['tags'].fillna('').astype(str) + ' ' + target_game['genres'].fillna('').astype(str))
        print(target_game_tags_and_genres)
        # Crea un vectorizador TF-IDF
        tfidf_vectorizer = TfidfVectorizer()

        
        similarity_scores = None

        # Procesa los juegos por lotes utilizando chunks
        
        # Combina las etiquetas (tags) y géneros de los juegos en una sola cadena de texto
        chunk_tags_and_genres = ' '.join(df_steam_games['tags'].fillna('').astype(str) + ' ' + df_steam_games['genres'].fillna('').astype(str))

        # Aplica el vectorizador TF-IDF al lote actual de juegos y al juego de referencia
        tfidf_matrix = tfidf_vectorizer.fit_transform([target_game_tags_and_genres, chunk_tags_and_genres])

        # Calcula la similitud entre el juego de referencia y los juegos del lote actual
        if similarity_scores is None:
            similarity_matrix = cosine_similarity(tfidf_matrix)
            similarity_scores = cosine_similarity(similarity_matrix, similarity_scores)
        else:
            similarity_matrix = cosine_similarity(tfidf_matrix)
            similarity_scores = cosine_similarity(similarity_matrix, similarity_scores)

        if similarity_scores is not None:
            # Obtiene los índices de los juegos más similares
            similar_games_indices = similarity_scores[0].argsort()[::-1]

            # Recomienda los juegos más similares (puedes ajustar el número de recomendaciones)
            num_recommendations = 5
            recommended_games = df_steam_games.loc[similar_games_indices[1:num_recommendations + 1]]
            print(recommended_games)
            # Devuelve la lista de juegos recomendados
            return recommended_games[['item_name','item_id']].to_dict(orient='records')

        return {"message": "No se encontraron juegos similares"}

    except:
        return {"message": f"Error: {str(traceback.format_exc())}"}
    

def recomendacion_juego2(product_id: str):
    try:
        # Obtener el juego objetivo
        df_steam_games['combined_tags_genres'] = df_steam_games['tags'].fillna('') + ' ' + df_steam_games['genres'].fillna('')
        target_game = df_steam_games[df_steam_games['item_name'].str.lower() == product_id.lower()]

        if target_game.empty:
            return {"message": "No se encontró el juego de referencia."}
        # Obtener la cadena de texto combinada del juego objetivo
        target_game_combined = target_game['combined_tags_genres'].iloc[0]
        
        # Añadir el juego objetivo al dataframe
        combined_tags_genres = [target_game_combined] + df_steam_games['combined_tags_genres'].tolist()

        # Crear un vectorizador TF-IDF y transformar los datos
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(combined_tags_genres)

        # Calcular la similitud del coseno entre el juego objetivo y todos los demás
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Obtener los índices de los juegos más similares
        similar_games_indices = similarity_matrix.argsort()[::-1]

        # Recomendar los juegos más similares (puedes ajustar el número de recomendaciones)
        num_recommendations = 5
        recommended_games = df_steam_games.iloc[similar_games_indices[:num_recommendations]]

        # Devolver la lista de juegos recomendados
        return recommended_games[['item_name', 'item_id']].to_dict(orient='records')

    except :
        return {"message": f"Error: {str(traceback.format_exc())}"}
