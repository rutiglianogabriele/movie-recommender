import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.data_loader import MovieLensDataLoader
from src.utils.preprocessing import (
    analyze_ratings_data,
    identify_sparse_entries,
    preprocess_ratings,
    analyze_rating_patterns,
    get_rating_sparsity
)
from src.config import Config


def main():
    """Main function to run the movie recommender preprocessing pipeline."""
    # Initialize configuration
    config = Config()
    
    # Create data loader instance
    data_loader = MovieLensDataLoader(config)
    
    # Load data
    print("Loading MovieLens dataset...")
    if not data_loader.load_data():
        print("Failed to load data. Please check the data files and paths.")
        return
    
    ratings_df = data_loader.ratings_df
    movies_df = data_loader.movies_df

    print(ratings_df.head())
    print(movies_df.head())
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")