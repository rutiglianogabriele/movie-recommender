# 🎬 Movie Recommender System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

A machine learning-based recommendation engine that suggests personalized movie recommendations based on user preferences and behavior patterns.

## ✨ Features

- 🎯 **Personalized Recommendations**: Delivers tailored movie suggestions based on individual user preferences and rating history
- 🧠 **Multiple Algorithm Support**: Implements both collaborative filtering and content-based recommendation approaches
- 📊 **Comprehensive Data Analysis**: Includes exploratory analysis of user behavior, rating patterns, and movie metadata
- 🔍 **Content Similarity**: Identifies movies with similar characteristics based on genres, actors, directors, and other metadata
- 📈 **Custom Evaluation Framework**: Measures recommendation quality using multiple metrics beyond standard error measurements

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Required libraries (install via pip):
  ```bash
  pip install -r requirements.txt
  ```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rutiglianogabriele/movie-recommender.git
   cd movie-recommender
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the MovieLens dataset:
   ```bash
   python src/data/download_data.py
   ```

## 📋 Usage

1. Run the exploratory data analysis notebook to understand the dataset:
   ```bash
   jupyter notebook notebooks/exploratory_data_analysis.ipynb
   ```

2. Preprocess the data and engineer features:
   ```bash
   python src/preprocessing/prepare_data.py
   ```

3. Train recommendation models:
   ```bash
   python src/models/train_models.py
   ```

4. Generate recommendations for users:
   ```bash
   python src/recommendation_engine.py --user_id 42
   ```

## 📂 Project Structure

```
movie-recommender/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                        # Original MovieLens dataset
│   └── processed/                  # Cleaned and transformed data
├── notebooks/
│   └── data_analysis.ipynb         # Initial data exploration
└── src/
    ├── data/
    |   ├── data_loader.py          # Helper functions to load the datasets into Pandas dataframes
    |   └── summary_statistics.py   # Helper functions for the data_analysis notebook
    ├── models/                     # Recommendation algorithms
    └── utils/
        └── preprocessing.py        # Helper functions for data pre-processing
```

## 🔧 Implementation Details

The recommendation system is implemented in multiple phases:

1. **Data Analysis**: Thorough exploration of rating distributions and user activity patterns
2. **Preprocessing**: Cleaning, normalization, and feature engineering
3. **Model Implementation**: Building and training different recommendation algorithms
4. **Evaluation**: Comparing model performance using various metrics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) for providing the dataset