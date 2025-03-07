import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ..config import Config

class MovieLensDataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.ratings_df = None
        self.movies_df = None
        
    def load_data(self):
        # Load MovieLens dataset from CSV files.
        try:
            self.ratings_df = pd.read_csv(
                self.config.DATA_DIR / self.config.RATINGS_FILE,
                usecols=['userId', 'movieId', 'rating', 'timestamp']
            )
            
            self.movies_df = pd.read_csv(
                self.config.DATA_DIR / self.config.MOVIES_FILE,
                usecols=['movieId', 'title', 'genres']
            )
            
            print(f"Loaded {len(self.ratings_df)} ratings and {len(self.movies_df)} movies.")
            return True
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self):
        # Preprocess the data by filtering and splitting into train/test sets.
        if self.ratings_df is None or self.movies_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Let's count the recurrences of each user and movie in the ratings DataFrame
        user_counts = self.ratings_df['userId'].value_counts()
        movie_counts = self.ratings_df['movieId'].value_counts()
        
        # Filter users and movies with less than MIN_RATINGS ratings
        valid_users = user_counts[user_counts >= self.config.MIN_RATINGS].index
        valid_movies = movie_counts[movie_counts >= self.config.MIN_RATINGS].index
        
        # Filter ratings DataFrame based on valid users and movies
        print(f"Number of rows in the DataFrame before preprocessing: {len(self.ratings_df)}")
        self.ratings_df = self.ratings_df[
            (self.ratings_df['userId'].isin(valid_users)) &
            (self.ratings_df['movieId'].isin(valid_movies))
        ]
        print(f"Number of rows in the DataFrame after preprocessing: {len(self.ratings_df)}")
        
        # Split into train and test sets
        train_data, test_data = train_test_split(
            self.ratings_df,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )
        
        return train_data, test_data
    
    def get_movie_features(self):
        # Extract movie features from genres.
        if self.movies_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Convert genres string to one-hot encoding
        genres_df = self.movies_df['genres'].str.get_dummies(sep='|')
        
        # Combine with movie information
        movie_features = pd.concat([
            self.movies_df[['movieId', 'title']],
            genres_df
        ], axis=1)
        
        return movie_features