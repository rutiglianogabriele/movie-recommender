from pathlib import Path

class Config:
    # Project structure
    ROOT_DIR = Path(__file__).resolve().parent.parent 
    DATA_DIR = ROOT_DIR / "src" / "data"
    MODELS_DIR = ROOT_DIR / "src" / "models"
    
    # Data configuration
    RATINGS_FILE = "ratings.csv"
    MOVIES_FILE = "movies.csv"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Model parameters
    MIN_RATINGS = 5  # Minimum ratings per user/movie
    K_NEIGHBORS = 20  # Number of neighbors for collaborative filtering