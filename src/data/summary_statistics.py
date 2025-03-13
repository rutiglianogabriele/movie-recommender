import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class MovieLensDataAnalyzer:
    def __init__(self) -> None:
        pass

    def analyze_data(self, ratings_df, movies_df):
        """Analyze ratings distribution and basic statistics."""
        analysis = {}
        
        # Basic statistics
        analysis['total_ratings'] = len(ratings_df)
        analysis['unique_users'] = ratings_df['userId'].nunique()
        analysis['total_reviewed_movies'] = ratings_df['movieId'].nunique()
        analysis['total_movies'] = movies_df['movieId'].nunique()
        analysis['average_rating'] = ratings_df['rating'].mean()
        analysis['rating_std'] = ratings_df['rating'].std()
        
        # Ratings distribution
        rating_counts = ratings_df['rating'].value_counts().sort_index()
        
        plt.figure(figsize=(10, 5))
        bars = plt.bar(rating_counts.index, rating_counts.values, 
                color='steelblue', width=0.4, edgecolor='black')
        
        # Add counts above each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1*max(rating_counts.values),
                    f'{height:,}', ha='center', va='bottom', fontsize=10)
        
        plt.title('Distribution of Ratings', fontsize=16)
        plt.xlabel('Rating Value', fontsize=14)
        plt.ylabel('Number of Ratings', fontsize=14)
        plt.xticks(sorted(ratings_df['rating'].unique()))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        """Analyze user activity and create segments."""
        # Count ratings per user
        user_activity = ratings_df['userId'].value_counts().reset_index()
        user_activity.columns = ['userId', 'rating_count']
        
        # Create user segments
        bins = [0, 5, 20, 50, 100, 200, 500, 1000, user_activity['rating_count'].max()]
        labels = ['1-5', '6-20', '21-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
        user_activity['segment'] = pd.cut(user_activity['rating_count'], bins=bins, labels=labels)
        
        # Count users in each segment
        segment_counts = user_activity['segment'].value_counts().sort_index()
        
        # Plot segments
        plt.figure(figsize=(10, 5))
        ax = segment_counts.plot(kind='bar', color='cornflowerblue')
        
        # Add count and percentage labels
        total_users = len(user_activity)
        for i, count in enumerate(segment_counts):
            percentage = 100 * count / total_users
            ax.text(i, count + 5, f'{count:,}\n({percentage:.1f}%)', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.title('User Segments by Rating Activity', fontsize=16)
        plt.xlabel('Number of Ratings Submitted', fontsize=14)
        plt.ylabel('Number of Users', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        return analysis

    def plot_popularity_vs_rating(self, ratings_df):
        """Analyze relationship between movie popularity and average rating."""
        # Calculate metrics per movie
        movie_stats = ratings_df.groupby('movieId').agg(
            num_ratings=('rating', 'count'),
            avg_rating=('rating', 'mean')
        ).reset_index()
        
        # Plot
        plt.figure(figsize=(10, 4))
        plt.scatter(movie_stats['num_ratings'], movie_stats['avg_rating'], 
                    alpha=0.5, edgecolors='none', c='steelblue')
        
        # Add trend line
        z = np.polyfit(np.log10(movie_stats['num_ratings']), movie_stats['avg_rating'], 1)
        p = np.poly1d(z)
        x_trend = np.logspace(0, np.log10(movie_stats['num_ratings'].max()), 100)
        plt.plot(x_trend, p(np.log10(x_trend)), 'r--', linewidth=2)
        
        plt.title('Movie Popularity vs. Average Rating', fontsize=16)
        plt.xlabel('Number of Ratings (log scale)', fontsize=14)
        plt.ylabel('Average Rating', fontsize=14)
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Calculate correlation
        correlation = np.corrcoef(np.log10(movie_stats['num_ratings']), movie_stats['avg_rating'])[0, 1]
        print(f"Correlation between log(popularity) and rating: {correlation:.3f}")
        
        return movie_stats
    
    def plot_rating_evolution(self,ratings_df):
        """Analyze how ratings distribution evolves over time."""
        # Convert timestamp to datetime
        ratings_df['date'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        
        # Extract year and month
        ratings_df['year'] = ratings_df['date'].dt.year
        
        # Only include years with sufficient data
        year_counts = ratings_df['year'].value_counts()
        valid_years = year_counts[year_counts > 1000].index
        filtered_df = ratings_df[ratings_df['year'].isin(valid_years)]
        
        # Calculate rating distribution per year
        grouped = filtered_df.groupby(['year', 'rating']).size().unstack()
        
        # Convert to percentages
        percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
        
        # Plot
        plt.figure(figsize=(14, 8))
        percentages.plot(kind='bar', stacked=True, colormap='viridis')
        
        plt.title('Rating Distribution Evolution Over Time', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Percentage of Ratings', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.legend(title='Rating Value')
        plt.tight_layout()
        plt.show()
        
        return percentages
    
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