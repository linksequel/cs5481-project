import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random
from tqdm import tqdm
from scipy.sparse.linalg import svds
import itertools

# Load the dataset
def load_lastfm(path="../datas/hetrec2011-lastfm-2k"):
    # Load user-artist interactions
    user_artists = pd.read_csv(f"{path}/user_artists.dat", sep='\t')
    
    # Load artists data
    artists = pd.read_csv(f"{path}/artists.dat", sep='\t')
    
    return user_artists, artists

# Preprocess data
def preprocess_data(user_artists):
    # Create user-item matrix
    user_item_matrix_df = user_artists.pivot(index='userID', columns='artistID', values='weight').fillna(0)
    return user_item_matrix_df

class CollaborativeFilteringRecommender:
    def __init__(self, user_item_matrix, n_users=10, n_items=10, n_factors=50, n_recommendations=10):
        self.user_item_matrix = user_item_matrix
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.n_recommendations = n_recommendations
    
    def user_based_cf(self, user_id):
        # Calculate user similarity
        user_similarity = cosine_similarity(self.user_item_matrix)
        user_similarity_df = pd.DataFrame(user_similarity, index=self.user_item_matrix.index, columns=self.user_item_matrix.index)
        
        # Find similar users
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:self.n_users+1].index
        
        # Get recommendations
        recommendations = defaultdict(float)
        
        for similar_user in similar_users:
            similarity_between_user = user_similarity_df.loc[user_id, similar_user]
            
            for item in self.user_item_matrix.columns:
                if self.user_item_matrix.loc[user_id, item] == 0 and self.user_item_matrix.loc[similar_user, item] > 0:
                    recommendations[item] += similarity_between_user * self.user_item_matrix.loc[similar_user, item]
        
        # Sort recommendations
        recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:self.n_recommendations]
        return recommendations
    
    def item_based_cf(self, user_id):
        # Calculate item similarity
        item_similarity = cosine_similarity(self.user_item_matrix.T)
        item_similarity_df = pd.DataFrame(item_similarity, index=self.user_item_matrix.columns, columns=self.user_item_matrix.columns)
        
        # Get items the user has interacted with
        user_related_items = self.user_item_matrix.loc[user_id]
        related_items_id = user_related_items[user_related_items > 0].index.tolist()
        
        # Get recommendations
        recommendations = defaultdict(float)
        
        for item_id in related_items_id:
            item_weight = self.user_item_matrix.loc[user_id, item_id]
            
            # Find similar items
            similar_items = item_similarity_df[item_id].sort_values(ascending=False)[1:self.n_items+1]
            
            for similar_item, similarity_between_item in similar_items.items():
                if similar_item in related_items_id:
                    continue
                recommendations[similar_item] += similarity_between_item * item_weight
        
        # Sort recommendations
        recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:self.n_recommendations]
        return recommendations
    
    def svd_based_cf(self, user_id):
        # Convert to numpy array for SVD
        user_item_array = self.user_item_matrix.values
        
        # Get user index
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        
        # Ensure n_factors is not too large
        n_factors = min(self.n_factors, min(user_item_array.shape) - 1)
        
        # Perform SVD
        u, sigma, vt = svds(user_item_array, k=n_factors)
        
        # Convert to diagonal matrix
        sigma_diag = np.diag(sigma)
        
        # Reconstruct the matrix using the factors
        reconstructed_matrix = np.dot(np.dot(u, sigma_diag), vt)
        
        # Get the reconstructed ratings for the user
        user_ratings = reconstructed_matrix[user_idx]
        
        # Get actual ratings for the user
        actual_ratings = user_item_array[user_idx]
        
        # Create a mask of items the user hasn't interacted with
        unrated_items = actual_ratings == 0
        
        # Get indices of items the user hasn't interacted with, ordered by predicted rating
        candidate_items = np.argsort(-user_ratings * unrated_items)[:self.n_recommendations]
        
        # Convert back to original item IDs and predicted scores
        recommendations = [(self.user_item_matrix.columns[item_idx], user_ratings[item_idx]) 
                          for item_idx in candidate_items if unrated_items[item_idx]]
        
        return recommendations
    
    def evaluate(self, test_ratio=0.2, n_sample_users=50):
        # Create a copy of the matrix to avoid modifying the original
        matrix = self.user_item_matrix.copy()
        
        # Metrics storage
        metrics = {
            'user_hr': [], 'user_ndcg': [], 'user_mrr': [],
            'item_hr': [], 'item_ndcg': [], 'item_mrr': [],
            'svd_hr': [], 'svd_ndcg': [], 'svd_mrr': []
        }
        
        sampled_users = random.sample(list(matrix.index), n_sample_users)
        print(f"Evaluating recommendations for {len(sampled_users)} users...")
        
        # For each user, hide some interactions as test data
        for user_id in tqdm(sampled_users, desc="Evaluating", ncols=80):
            # Get items this user has interacted with
            user_items = matrix.columns[matrix.loc[user_id] > 0].tolist()
            
            # Skip users with too few interactions
            if len(user_items) <= 2:
                continue
                
            # Randomly select items for testing
            n_test = max(1, int(len(user_items) * test_ratio))
            test_items = random.sample(user_items, n_test)
            
            # Create training matrix by setting test items to zero
            train_matrix = matrix.copy()
            for item in test_items:
                train_matrix.loc[user_id, item] = 0
            
            # Create a temporary recommender with the training matrix but same hyperparameters
            temp_recommender = CollaborativeFilteringRecommender(
                train_matrix, 
                self.n_users, 
                self.n_items, 
                self.n_factors, 
                self.n_recommendations
            )
            
            # Get recommendations from all methods
            user_recs = temp_recommender.user_based_cf(user_id)
            item_recs = temp_recommender.item_based_cf(user_id)
            svd_recs = temp_recommender.svd_based_cf(user_id)
            
            # Extract just the item IDs
            user_rec_items = [item_id for item_id, _ in user_recs]
            item_rec_items = [item_id for item_id, _ in item_recs]
            svd_rec_items = [item_id for item_id, _ in svd_recs]
            
            # Calculate metrics
            metrics['user_hr'].append(hit_ratio(user_rec_items, test_items))
            metrics['user_ndcg'].append(ndcg(user_rec_items, test_items))
            metrics['user_mrr'].append(mrr(user_rec_items, test_items))
            
            metrics['item_hr'].append(hit_ratio(item_rec_items, test_items))
            metrics['item_ndcg'].append(ndcg(item_rec_items, test_items))
            metrics['item_mrr'].append(mrr(item_rec_items, test_items))
            
            metrics['svd_hr'].append(hit_ratio(svd_rec_items, test_items))
            metrics['svd_ndcg'].append(ndcg(svd_rec_items, test_items))
            metrics['svd_mrr'].append(mrr(svd_rec_items, test_items))
        
        # Calculate average metrics
        avg_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
        
        return avg_metrics

def evaluate_hyperparameters(user_item_matrix, param_grid, test_ratio=0.2, n_sample_users=50):
    """
    Evaluate multiple hyperparameter combinations and return results.
    
    Args:
        user_item_matrix: The user-item matrix
        param_grid: Dictionary of parameter lists to test (e.g., {'n_users': [5, 10, 20]})
        test_ratio: Ratio of data to use for testing
        n_sample_users: Number of users to sample for evaluation
        
    Returns:
        List of dictionaries containing parameters and metrics
    """
    results = []
    
    # Generate all combinations of parameters
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    for params in tqdm(param_combinations, desc="Testing hyperparameters", ncols=80):
        param_dict = dict(zip(param_keys, params))
        
        # Create recommender with current parameters
        recommender = CollaborativeFilteringRecommender(
            user_item_matrix,
            n_users=param_dict.get('n_users', 10),
            n_items=param_dict.get('n_items', 10),
            n_factors=param_dict.get('n_factors', 50),
            n_recommendations=param_dict.get('n_recommendations', 10)
        )
        
        # Evaluate with current parameters
        metrics = recommender.evaluate(test_ratio, n_sample_users)
        
        # Store results
        result = {**param_dict, **metrics}
        results.append(result)
    
    return results

def display_results_table(results):
    """Display results in a table format."""
    df = pd.DataFrame(results)
    
    # Reorder columns to group parameters first, then metrics by method
    param_cols = [col for col in df.columns if col.startswith('n_')]
    metric_cols = [col for col in df.columns if not col.startswith('n_')]
    
    df = df[param_cols + metric_cols]
    
    # Round metrics to 4 decimal places for better display
    for col in metric_cols:
        df[col] = df[col].round(4)
    
    # Set pandas display options to show all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.expand_frame_repr', False)
    
    return df

# Hit Ratio@K
def hit_ratio(recommended_items, test_items):
    hits = len(set(recommended_items) & set(test_items))
    return hits / len(test_items) if test_items else 0

# NDCG@K
def ndcg(recommended_items, test_items):
    dcg = 0
    idcg = 0
    
    # Calculate DCG
    for i, item in enumerate(recommended_items):
        if item in test_items:
            # Using binary relevance (1 if hit, 0 if miss)
            dcg += 1 / np.log2(i + 2)  # i+2 because i starts from 0
    
    # Calculate IDCG (ideal DCG - items are perfectly ranked)
    for i in range(min(len(test_items), len(recommended_items))):
        idcg += 1 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0

# MRR@K
def mrr(recommended_items, test_items):
    for i, item in enumerate(recommended_items):
        if item in test_items:
            return 1 / (i + 1)  # i+1 because i starts from 0
    return 0

def load_and_preprocess_data():
    user_artists, artists = load_lastfm()
    user_item_matrix_df = preprocess_data(user_artists)
    return user_artists, artists, user_item_matrix_df

def print_recommendations(user_id, recommendations, artists, method_name):
    print(f"\n{method_name} recommendations for user {user_id}:")
    for item_id, score in recommendations:
        artist_name = artists[artists['id'] == item_id]['name'].values[0] if item_id in artists['id'].values else "Unknown"
        print(f"Artist ID: {item_id}, Score: {score:.2f}, Name: {artist_name}")

def main():
    # Load and preprocess data
    user_artists, artists, user_item_matrix_df = load_and_preprocess_data()
    
    # Example: Test different hyperparameter combinations
    param_grid = {
        'n_users': [10],
        'n_items': [10],
        'n_factors': [50],
        'n_recommendations': [10]
    }
    
    print("Testing different hyperparameter combinations...")
    results = evaluate_hyperparameters(user_item_matrix_df, param_grid, n_sample_users=30)
    
    # Display results in a table
    results_df = display_results_table(results)
    print("\nHyperparameter Evaluation Results:")
    print(results_df)
    
    # You might want to save results to a CSV for further analysis
    results_df.to_csv("lastfm.csv", index=False)
    
    # Find the best parameters for each method
    print("\nBest hyperparameters by method:")
    best_user_based = results_df.loc[results_df['user_hr'].idxmax()]
    best_item_based = results_df.loc[results_df['item_hr'].idxmax()]
    best_svd_based = results_df.loc[results_df['svd_hr'].idxmax()]
    
    print(f"User-based CF: {best_user_based[['n_users', 'n_items', 'n_factors', 'n_recommendations']].to_dict()}")
    print(f"Item-based CF: {best_item_based[['n_users', 'n_items', 'n_factors', 'n_recommendations']].to_dict()}")
    print(f"SVD-based CF: {best_svd_based[['n_users', 'n_items', 'n_factors', 'n_recommendations']].to_dict()}")

if __name__ == "__main__":
    main()