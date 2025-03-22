from pathlib import Path

class Config:
    # Project structure
    ROOT_DIR = Path(__file__).resolve().parent.parent 
    DATA_DIR = ROOT_DIR / "data"
    MODELS_DIR = ROOT_DIR / "src" / "models"
    NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
    DARA_LOAD_DIR = ROOT_DIR / "src" / "data"
    
    
    # Data configuration
    RATINGS_FILE = "ratings.csv"
    MOVIES_FILE = "movies.csv"