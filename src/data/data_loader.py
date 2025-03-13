import pandas as pd
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
