from pathlib import Path

class Config:
    # Project structure
    ROOT_DIR = Path(__file__).resolve().parent.parent 
    DATA_DIR = ROOT_DIR / "src" / "data"
    MODELS_DIR = ROOT_DIR / "src" / "models"
    
    # Data configuration
    RATINGS_FILE = "ratings.csv"
    MOVIES_FILE = "movies.csv"