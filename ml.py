import pandas as pd

credit_df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Pictures\Desktop\Innomatics_workspace\Fast_API_Streamlit_project\Datasets\movie dataset\credits.csv')
keyword_df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Pictures\Desktop\Innomatics_workspace\Fast_API_Streamlit_project\Datasets\movie dataset\keywords.csv')
links_small_df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Pictures\Desktop\Innomatics_workspace\Fast_API_Streamlit_project\Datasets\movie dataset\links_small.csv')
links_df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Pictures\Desktop\Innomatics_workspace\Fast_API_Streamlit_project\Datasets\movie dataset\links.csv')
movie_metadata_df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Pictures\Desktop\Innomatics_workspace\Fast_API_Streamlit_project\Datasets\movie dataset\movies_metadata.csv',low_memory=False)
ratings_small_df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Pictures\Desktop\Innomatics_workspace\Fast_API_Streamlit_project\Datasets\movie dataset\ratings_small.csv')
ratings_df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Pictures\Desktop\Innomatics_workspace\Fast_API_Streamlit_project\Datasets\movie dataset\ratings.csv')

# What low_memory=False does:

# Forces pandas to read the entire file at once
# Makes a single correct dtype decision per column
# Avoids mixed type conflicts
print(movie_metadata_df.columns)
# Handling null values
print(movie_metadata_df.isna().sum())

movie_metadata_df = movie_metadata_df.drop (columns=['belongs_to_collection','homepage','poster_path','tagline'])
movie_metadata_df.dropna(subset=['title'],inplace = True)
movie_metadata_df.dropna(subset=['imdb_id'],inplace=True)
# Convert all these columns to numeric safely
col_to_numeric = ['popularity','revenue','runtime','vote_average','vote_count','budget']
for col in col_to_numeric:
    movie_metadata_df[col] = pd.to_numeric(movie_metadata_df[col],errors='coerce')

movie_metadata_df['overview']=movie_metadata_df['overview'].fillna('',inplace=True)
movie_metadata_df['runtime']=movie_metadata_df['runtime'].fillna(movie_metadata_df['runtime'].median(),inplace=True)
movie_metadata_df['vote_average']=movie_metadata_df['vote_average'].fillna(movie_metadata_df['vote_average'].mean(),inplace=True)
movie_metadata_df['vote_count']=movie_metadata_df['vote_count'].fillna(movie_metadata_df['vote_count'].median(),inplace=True)
movie_metadata_df['popularity']=movie_metadata_df['popularity'].fillna(movie_metadata_df['popularity'].median(),inplace=True)
movie_metadata_df['revenue']=movie_metadata_df['revenue'].fillna(0,inplace=True)
# 2. Fill original_language nulls with mode (most frequent language)
movie_metadata_df['original_language']=movie_metadata_df['original_language'].fillna(movie_metadata_df['original_language'].mode()[0])

# 3. Fill overview nulls with empty string
movie_metadata_df['overview']=movie_metadata_df['overview'].fillna('')

# 4. Fill release_date nulls with mode
movie_metadata_df['release_date']=movie_metadata_df['release_date'].fillna(movie_metadata_df['release_date'].mode()[0], inplace=True)

# 5. Fill runtime nulls with median
movie_metadata_df['runtime']=movie_metadata_df['runtime'].fillna(movie_metadata_df['runtime'].median(), inplace=True)

# 6. Fill status nulls with mode (most frequent status)
movie_metadata_df['status']=movie_metadata_df['status'].fillna(movie_metadata_df['status'].mode()[0], inplace=True)


print(movie_metadata_df.isna().sum())
# 10. Drop rows where tmdbId is null (13 rows)
links_small_df.dropna(subset=['tmdbId'], inplace=True)

# ── links ─────────────────────────────────────────────────────────────

# 11. Drop rows where tmdbId is null (219 rows)
links_df.dropna(subset=['tmdbId'], inplace=True)

print("links_small nulls remaining:", links_small_df.isnull().sum().sum())
print("links nulls remaining:", links_df.isnull().sum().sum())


print(ratings_small_df.head())

ratings_small_df['target'] = (ratings_small_df['rating'] >= 4.0).astype(int)

print(ratings_small_df['target'])

# Fix dtype mismatch before merge
links_small_df['tmdbId'] = pd.to_numeric(links_small_df['tmdbId'], errors='coerce')
links_small_df.dropna(subset=['tmdbId'], inplace=True)
links_small_df['tmdbId'] = links_small_df['tmdbId'].astype(int)

movie_metadata_df['id'] = pd.to_numeric(movie_metadata_df['id'], errors='coerce')
movie_metadata_df.dropna(subset=['id'], inplace=True)
movie_metadata_df['id'] = movie_metadata_df['id'].astype(int)

# merge ratings with links to get tmdbId
merged_df = ratings_small_df.merge(links_small_df, on='movieId', how='inner')

# merge with movies_metadata to get movie features
merged_df = merged_df.merge(movie_metadata_df, left_on='tmdbId', right_on='id', how='inner')
print('\n Merged df: \n',merged_df.head())

# Single bracket [] → returns a Series (1D)
x = merged_df[['popularity', 'vote_average', 'vote_count', 'runtime', 'revenue', 'budget',
                'original_language', 'status']]
# Double bracket [[]] → returns a DataFrame (2D)
y = merged_df['target']

numerical_col = x.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_col = x.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical columns: \n{numerical_col}")
print(f"\nCategorical columns: \n{categorical_col}")


########## PIPELINE START ###########

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


numeric_pipeline = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])

categoric_pipeline = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(sparse_output=False,handle_unknown='ignore'))
])


final_pipeline = ColumnTransformer(transformers=[
    ('numeric',numeric_pipeline,numerical_col),
    ('catagoric',categoric_pipeline,categorical_col)
])

model = Pipeline(steps=[
    ('pipeline',final_pipeline),
    ('model',XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1,
        random_state=42,
        eval_metric='logloss'
    ))
])

# Classification Problem  →  stratify=y  ✅
# Regression Problem      →  stratify=y  ❌
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2,stratify=y)

model.fit(x_train,y_train)

pred = model.predict(x_test)


from sklearn.metrics import accuracy_score,classification_report

print(f"\nAccuracy score:{accuracy_score(y_test,pred)}\n")

print(f"\nAccuracy score:\n{classification_report(y_test,pred)}")

import joblib

joblib.dump(model,'movie_recommendation_model.pkl')
print('model saved successfully')

# Load and test
loaded_model = joblib.load('movie_recommendation_model.pkl')
test_pred = loaded_model.predict(x_test)
print("Loaded Model Accuracy:", accuracy_score(y_test, test_pred))



