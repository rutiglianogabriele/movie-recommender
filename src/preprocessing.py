import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def analyze_ratings_data(ratings_df):
    """Analyze ratings distribution and basic statistics."""
    analysis = {}
    
    # Basic statistics
    analysis['total_ratings'] = len(ratings_df)
    analysis['unique_users'] = ratings_df['userId'].nunique()
    analysis['unique_movies'] = ratings_df['movieId'].nunique()
    analysis['average_rating'] = ratings_df['rating'].mean()
    analysis['rating_std'] = ratings_df['rating'].std()
    
    # Ratings distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=ratings_df, x='rating', bins=10)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()
    
    # User activity analysis
    user_ratings_count = ratings_df['userId'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.histplot(user_ratings_count, bins=50)
    plt.title('Distribution of Ratings per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.show()
    
    analysis['ratings_per_user_mean'] = user_ratings_count.mean()
    analysis['ratings_per_user_median'] = user_ratings_count.median()
    
    return analysis

def identify_sparse_entries(ratings_df, min_ratings_user=10, min_ratings_movie=10):
    """Identify users and movies with too few ratings."""
    user_ratings_count = ratings_df['userId'].value_counts()
    movie_ratings_count = ratings_df['movieId'].value_counts()
    
    sparse_users = user_ratings_count[user_ratings_count < min_ratings_user].index
    sparse_movies = movie_ratings_count[movie_ratings_count < min_ratings_movie].index
    
    return sparse_users, sparse_movies

def preprocess_ratings(ratings_df, min_ratings_user=10, min_ratings_movie=10):
    """Preprocess ratings data for ML algorithm."""
    # Remove sparse entries
    sparse_users, sparse_movies = identify_sparse_entries(
        ratings_df, 
        min_ratings_user, 
        min_ratings_movie
    )
    
    filtered_df = ratings_df[
        (~ratings_df['userId'].isin(sparse_users)) & 
        (~ratings_df['movieId'].isin(sparse_movies))
    ]
    
    # Center ratings around mean
    user_mean_ratings = filtered_df.groupby('userId')['rating'].mean()
    centered_ratings = pd.merge(
        filtered_df,
        user_mean_ratings.to_frame('user_mean'),
        left_on='userId',
        right_index=True
    )
    centered_ratings['rating_centered'] = (
        centered_ratings['rating'] - centered_ratings['user_mean']
    )
    
    # Check for rating bias
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=filtered_df, x='rating')
    plt.title('Rating Distribution After Filtering')
    plt.show()
    
    return filtered_df, centered_ratings

def analyze_rating_patterns(ratings_df):
    """Analyze temporal patterns and rating biases."""
    # Convert timestamp to datetime if needed
    if 'timestamp' in ratings_df.columns:
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        
        # Temporal analysis
        ratings_df['hour'] = ratings_df['timestamp'].dt.hour
        ratings_df['day_of_week'] = ratings_df['timestamp'].dt.day_name()
        
        plt.figure(figsize=(12, 5))
        sns.boxplot(data=ratings_df, x='day_of_week', y='rating')
        plt.title('Rating Distribution by Day of Week')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # Rating bias by user
    user_avg_ratings = ratings_df.groupby('userId')['rating'].agg(['mean', 'std'])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=user_avg_ratings, x='mean', y='std')
    plt.title('User Rating Bias: Mean vs Standard Deviation')
    plt.xlabel('Average Rating')
    plt.ylabel('Rating Standard Deviation')
    plt.show()
    
    return user_avg_ratings

def get_rating_sparsity(ratings_df):
    """Calculate the sparsity of the user-item matrix."""
    n_users = ratings_df['userId'].nunique()
    n_movies = ratings_df['movieId'].nunique()
    n_ratings = len(ratings_df)
    
    sparsity = 1 - (n_ratings / (n_users * n_movies))
    return sparsity