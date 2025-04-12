import os
from pathlib import Path

class Config:
    def __init__(self):
        self.BASE_DIR = Path(os.path.dirname(os.path.dirname(__file__)))
        self.DATA_DIR = self.BASE_DIR / 'data'
        self.RATINGS_FILE = 'ratings.csv'
        self.MOVIES_FILE = 'movies.csv'