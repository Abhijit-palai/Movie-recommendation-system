import streamlit as st
import requests

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon = "🎬",
    layout= "centered"
)

st.title("Movie Recommendation System")

st.write("Fill in the movie details to get a recommendation")

popularity = st.number_input("Popularity ",min_value=0.0,value=21.9)
vote_average = st.number_input("Vote Average",min_value=0.0,max_value=10.0,value=7.8)
vote_count = st.number_input("Vote count",min_value=0,value= 1000)
runtime = st.number_input("Runtime (in minutes)",min_value=0.0,value=120.0)
revenue = st.number_input("Revenue",min_value=0.0,value=500000.0)
budget = st.number_input("Budget",min_value=0.0,value= 100000.0)
original_language = st.selectbox("Original Language",['en','fr','hi','ja','de','es','it','ko'])
status = st.selectbox("Status",["Released","Post Production","Rumored","In Production"])



# predict Button

if st.button("Get Recommendation"):
    # prepare data 
    input_data = {
        'popularity':popularity,
        'vote_average':vote_average,
        'vote_count':vote_count,
        'runtime':runtime,
        'revenue':revenue,
        'budget':budget,
        'original_language':original_language,
        'status':status
    }
    
    response = requests.post("http://localhost:8000/predict",json=input_data)
    
    if response.status_code == 200:
        result = response.json()
        prediction = result['prediction']
        recommendation = result['recommendation']
        
        if prediction == 1:
            st.success(f"{recommendation}")
            st.balloons()
        else:
            st.error(f"{recommendation}")
    else:
        st.error("Something went wrong ! Check in FastAPI")
        
    

