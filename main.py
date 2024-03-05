import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from fastapi import FastAPI

app = FastAPI()

# Load Dataset
df_rating = pd.read_csv('./Data/Dataset.csv')
df_title = pd.read_csv('./Data/Movie_Id_Titles.csv')

# Drop unnecessary columns
df_rating.drop(columns=['timestamp'], inplace=True)

# Merge datasets
merged_df = pd.merge(df_rating, df_title, on='item_id', how='inner')

# Create user-item matrix
user_item_matrix = pd.pivot_table(merged_df, values='rating', index='user_id', columns='item_id', fill_value=0)

# Split data into training and testing sets
train_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)

# Matrix factorization using Singular Value Decomposition (SVD)
def matrix_factorization(train_matrix, num_latent_factors=10):
    U, sigma, Vt = np.linalg.svd(train_matrix, full_matrices=False)
    U = U[:, :num_latent_factors]
    sigma = np.diag(sigma[:num_latent_factors])
    Vt = Vt[:num_latent_factors, :]
    return U, sigma, Vt

# Predict ratings
U, sigma, Vt = matrix_factorization(train_data)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)

def get_movie_title(movie_id):
    return merged_df.loc[merged_df['item_id'] == movie_id, 'title'].values[0]

def process_recommendations(user_ratings, top_n_recommendations):
    recommendations = []
    for idx in top_n_recommendations:
        movie_id = user_item_matrix.columns[idx]
        movie_title = get_movie_title(movie_id)
        estimated_rating = user_ratings[idx]
        recommendation_info = {"movie_title": movie_title, "estimated_rating": estimated_rating}
        recommendations.append(recommendation_info)
        print(f"{movie_title} (Estimated Rating: {estimated_rating:.2f})")
    return recommendations

@app.get("/recommendations/")
def get_recommendations(user_id: int, top_n: int=5):
    user_ratings = predicted_ratings[user_id - 1]  # Adjust for 0-based indexing
    top_n_recommendations = np.argsort(user_ratings)[::-1][:top_n]

    recommendations = process_recommendations(user_ratings, top_n_recommendations)
    
    return {"user_id": user_id, "recommendations": recommendations}
