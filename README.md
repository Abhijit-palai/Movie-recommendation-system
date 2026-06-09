
End-to-end Movie Recommendation System using XGBoost + FastAPI + Streamlit | MovieLens Dataset | Binary Classification
# 🎬 Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange)

## 📌 Overview
A complete end-to-end Machine Learning project that 
predicts whether a user will LIKE ✅ or DISLIKE ❌ 
a movie based on movie features using XGBoost 
Classifier served via FastAPI and visualized 
through an interactive Streamlit UI.

---

## 🎯 Problem Statement
With thousands of movies available, users struggle 
to find movies they will enjoy. This system solves 
that by predicting movie preferences based on 
movie metadata features like popularity, ratings, 
runtime, budget and revenue.

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| API Backend | FastAPI, Uvicorn |
| Frontend UI | Streamlit |
| Model Saving | Joblib |
| Version Control | Git, GitHub |

---

## 📂 Dataset
MovieLens Dataset containing:

| File | Description | Rows |
|------|-------------|------|
| ratings_small.csv | User movie ratings | 100,004 |
| movies_metadata.csv | Movie details | 45,466 |
| links_small.csv | MovieLens to TMDB ID mapping | 9,125 |
| keywords.csv | Movie keywords | 46,419 |
| credits.csv | Cast and crew | 45,476 |
| links.csv | Full ID mapping | 45,843 |
| ratings.csv | Full ratings dataset | 26M+ |

---

## 🔄 Complete Project Flow

### 1️⃣ Data Loading
- Loaded 7 CSV files using Pandas
- Used low_memory=False for movies_metadata

### 2️⃣ Data Preprocessing
- Dropped irrelevant columns
- Handled null values with median/mode/mean
- Converted mixed dtypes to numeric
- Merged datasets using MovieLens → TMDB ID bridge

### 3️⃣ Feature Engineering
- Created binary target column (rating >= 4.0 → 1)
- Selected numerical and categorical features
- Built Scikit-learn Pipeline with:
  - Numerical → SimpleImputer + StandardScaler
  - Categorical → SimpleImputer + OneHotEncoder
  - ColumnTransformer for combining both

### 4️⃣ Model Training
- Algorithm: XGBoost Classifier
- Problem Type: Binary Classification
- Train/Test Split: 80/20 with stratify=y
- Manual Hyperparameter Tuning

### 5️⃣ Model Evaluation
- Accuracy: 65%+
- Metrics: Precision, Recall, F1-Score
- Classification Report

### 6️⃣ Model Saving
- Saved complete pipeline using Joblib

### 7️⃣ FastAPI Backend
- REST API with POST /predict endpoint
- Pydantic input validation
- Returns prediction + recommendation

### 8️⃣ Streamlit Frontend
- Interactive input form
- Real time predictions
- Visual result display

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 65% |
| Precision | 65% |
| Recall | 65% |
| F1 Score | 65% |

---

## 🚀 Features
- ✅ Complete ML Pipeline
- ✅ REST API with FastAPI
- ✅ Interactive UI with Streamlit
- ✅ Binary Classification
- ✅ Real time Predictions
- ✅ Industrial Project Structure

---

## ⚙️ Installation & Setup

### 1. Clone the repository
git clone https://github.com/Abhijit-palai/movie-recommendation-system.git
cd movie-recommendation-system

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Download Dataset
Download MovieLens dataset from:
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
Place CSV files in data/raw/ folder

### 5. Train the model
python ml.py

### 6. Start FastAPI server
uvicorn api.fast:app --reload

### 7. Start Streamlit app
streamlit run frontend/app.py

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | Health check |
| /predict | POST | Get movie recommendation |
| /docs | GET | Swagger UI |

### Sample API Request:
{
    "popularity": 21.9,
    "vote_average": 7.8,
    "vote_count": 1000,
    "runtime": 120.0,
    "revenue": 500000,
    "budget": 100000,
    "original_language": "en",
    "status": "Released"
}

### Sample API Response:
{
    "prediction": 1,
    "recommendation": "✅ Liked"
}


---

## 🔮 Future Improvements
- [ ] Add Collaborative Filtering
- [ ] Add Content Based Filtering
- [ ] Include genres and keywords features
- [ ] Deploy on AWS/GCP/Azure
- [ ] Add user authentication
- [ ] Add movie poster display
- [ ] Improve accuracy with deep learning

---

## 👤 Author
**Abhijit Palai**
- LinkedIn: linkedin.com/in/abhijitpalai/
- GitHub: github.com/Abhijit-palai
- Email: pabhijit2021@gmail.com

---

## 📄 License
This project is licensed under the MIT License

---

⭐ If you found this project helpful, please give it a star!
