from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd
import numpy as np

model = joblib.load(r'C:\Users\ASUS\OneDrive\Pictures\Desktop\Innomatics_workspace\Fast_API_Streamlit_project\movie_recommendation_model.pkl')

# Golden Rule — column names in input_df during prediction must exactly match column names used during training — same name, same order!
class validation_data(BaseModel):
    popularity:float
    vote_average:float
    vote_count:float
    runtime:float
    revenue:float
    budget:float
    original_language:str
    status:str


app = FastAPI(
    title="Movie Recommendation System",
    description = "XGBoost based movie recommendation API",
    version = "1.0.0"
)


@ app.get('/')
def home():
    return{
        'message':'The movie recommendation system is running....'
    }


# prediction endpoint

# model is not stored in the request — it is loaded on the server. You are sending input data via POST to the server, server processes it through the model and returns prediction back to you!

@app.post('/predict')
def predict(data:validation_data):
    
    input_df= pd.DataFrame([{
        'popularity':data.popularity,
        'vote_average':data.vote_average,
        'vote_count':data.vote_count,
        'runtime':data.runtime,
        'revenue':data.revenue,
        'budget':data.budget,
        'original_language':data.original_language,
        'status':data.status
    }])
    
    # predict 
    
    prediction = model.predict(input_df)[0] # only take the first prediction.
    
    return {
        "prediction":int(prediction),
        "recommendation":"Liked" if prediction == 1 else "Not Liked"
    }
    
    
    