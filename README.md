<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

<p align="center">
<img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"  height=300>
</p>

# Descripcion del proyecto
Para este proyecto se nos pidio una limpieza de datos de varios datasets de la plataforma **Steam**, el objetivo es lograr una organizacion adecuada de los datos para trabajar sobre ellos
y conseguir un **MVP**.
Desarrollamos el rol de **Data Engineer** para poner a prueba los conocimientos adquiridos durante la carrera.

# Objetivos
***'sentiment_analysis'***: Crear una columna para identificar el analisis de sentimientos donde los valores son: '0' si es malo, '1' si es neutral y '2' si es positivo.

***'API'***: Crear una api consumible para acceder a los datos de forma correcta, los endpoints son los siguientes:

+ def **developer( *`desarrollador` : str* )**:
    `Cantidad` de items y `porcentaje` de contenido Free por año según empresa desarrolladora.

  + def **userdata( *`User_id` : str* )**:
    Debe devolver `cantidad` de dinero gastado por el usuario, el `porcentaje` de recomendación en base a reviews.recommend y `cantidad de items`.

    + def **UserForGenre( *`genero` : str* )**:
    Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.

+ def **best_developer_year( *`año` : int* )**:
   Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos)

  + def **developer_reviews_analysis( *`desarrolladora` : str* )**:
    Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total 
    de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.

El deployment de la API se genera en **Render**(https://render.com/docs/free#free-web-services)

Una vez los datos sean consumidos por la API se procede a implementar un sistema de recomendacion usando **Machine Learning**. El sistema es un filtro item-item, en el cual se toma un item y en base a ese item el modelo recomienda otro item similar usando **similitud del coseno**.


Los datasets originales y el diccionario de datos se encuentran en: [https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj]

Links del proyecto terminado:
                          Repositorio Github: [https://github.com/PierottiBR/PI_N1_MLOPS]
                          Deployment Render:[https://pi-n1-mlops.onrender.com]

