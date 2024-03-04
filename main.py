import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from fastapi import FastAPI
app = FastAPI()

# Load Dataset
df_rating = pd.read_csv(r'C:\Users\User\Desktop\Movie-Recommendation-System\Data\Dataset.csv')
df_title = pd.read_csv(r'C:\Users\User\Desktop\Movie-Recommendation-System\Data\Movie_Id_Titles.csv')

# Drop unnecessary columns
df_rating.drop(columns=['timestamp'], inplace=True)

# Merge datasets
merged_df = pd.merge(df_rating, df_title, on='item_id', how='inner')

# Create user-item matrix
user_item_matrix = pd.pivot_table(merged_df, values='rating', index='user_id', columns='item_id', fill_value=0)

# Split data into training and testing sets
train_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)

# Matrix factorization using Singular Value Decomposition (SVD)
U, sigma, Vt = np.linalg.svd(train_data, full_matrices=False)

# Choose the number of latent factors (adjust as needed)
num_latent_factors = 10

# Reduce dimensions
U = U[:, :num_latent_factors]
sigma = np.diag(sigma[:num_latent_factors])
Vt = Vt[:num_latent_factors, :]

# Predict ratings
predicted_ratings = np.dot(np.dot(U, sigma), Vt)

@app.get("/recommendations/")
def get_recommendations(user_id: int):
    user_ratings = predicted_ratings[user_id - 1]  # Adjust for 0-based indexing
    top_n_recommendations = np.argsort(user_ratings)[::-1][:10]

    recommendations = []
    for idx in top_n_recommendations:
        movie_id = user_item_matrix.columns[idx]
        movie_title = merged_df.loc[merged_df['item_id'] == movie_id, 'title'].values[0]
        estimated_rating = user_ratings[idx]
        recommendation_info = {"movie_title": movie_title, "estimated_rating": estimated_rating}
        recommendations.append(recommendation_info)
        print(f"{movie_title} (Estimated Rating: {estimated_rating:.2f})")

    return {"user_id": user_id, "recommendations": recommendations}