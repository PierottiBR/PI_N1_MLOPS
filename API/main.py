from fastapi import FastAPI
from funciones import *
import json 

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "API TEST"}

@app.get("/user_for_genre")
async def user_for_genre(genero):
    return UserForGenre(genero)

@app.get('/user_data')
async def user_data(user_id):
    return userdata(user_id)

@app.get('/developer_data')
async def developer_data(developerdata):
    return json.loads(developer(developerdata))

@app.get('/developer_of_year')
async def developer_of_year(year):
    return best_developer_year(year)

@app.get('/developer_reviews_analysis')
async def developer_reviews_analysis(desarrolladora):
    return reviews_analysis(desarrolladora)
    
@app.get('/Recomendation_System')
async def RecomendationSystem(title: str):
    return recomendacion_juego2(title)