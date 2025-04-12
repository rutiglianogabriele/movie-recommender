import pandas as pd
from pathlib import Path

class MovieLensDataLoader:
    def __init__(self, config):
        self.data_dir = config.DATA_DIR
        self.ratings_file = config.RATINGS_FILE
        self.movies_file = config.MOVIES_FILE
        self.ratings_df = None
        self.movies_df = None
    
    def load_data(self):
        """Load MovieLens dataset from CSV files."""
        ratings_path = self.data_dir / self.ratings_file
        movies_path = self.data_dir / self.movies_file
        
        self.ratings_df = pd.read_csv(ratings_path)
        self.movies_df = pd.read_csv(movies_path)
        
        print(f"Successfully loaded {len(self.ratings_df)} ratings and {len(self.movies_df)} movies.")
        