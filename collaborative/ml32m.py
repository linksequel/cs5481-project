import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

"""
author: suqiulin-72405483
- Perform collaborative recommendation algorithm(usercf/svd) on this dataset
    - itemCF won't be implement because there are 292757 movies(item) which is very slow performed on traditional cpu
"""
# Load the ML-32M dataset
def load_ml32m(path="./datas/ml-32m"):
    print("Loading ML-32M dataset...")
    # Load ratings data in chunks due to large size
    ratings = pd.read_csv(f"{path}/ratings.csv", chunksize=1000000)
    ratings_df = pd.concat(ratings)
    # Optional: load movie data if available
    try:
        movies = pd.read_csv(f"{path}/movies.csv")
    except:
        movies = None
        print("Movies data not found. Will continue without movie titles.")
    
    return ratings_df, movies

# Preprocess data
def preprocess_data(ratings_df, sample_size=None):
    print("Preprocessing data...")
    if sample_size:
        # Take a sample if full dataset is too large
        user_counts = ratings_df['userId'].value_counts()
        users_to_keep = user_counts[user_counts >= 5].index[:sample_size]
        ratings_df = ratings_df[ratings_df['userId'].isin(users_to_keep)]
    
    # Create sparse user-item matrix
    users = ratings_df['userId'].unique()
    items = ratings_df['movieId'].unique()
    
    user_mapper = {user: i for i, user in enumerate(users)}
    item_mapper = {item: i for i, item in enumerate(items)}
    
    user_indices = [user_mapper[user] for user in ratings_df['userId']]
    item_indices = [item_mapper[item] for item in ratings_df['movieId']]
    
    # Create sparse matrix
    matrix = csr_matrix((ratings_df['rating'], (user_indices, item_indices)), 
                        shape=(len(users), len(items)))
    
    return matrix, users, items, user_mapper, item_mapper

# User-based collaborative filtering
def user_based_cf(matrix, user_idx, n_users=10, n_recommendations=10):
    # Get user row
    user_row = matrix[user_idx].toarray().flatten()
    
    # Calculate similarities
    similarities = []
    for i in range(matrix.shape[0]):
        if i != user_idx:
            other_row = matrix[i].toarray().flatten()
            # Compute similarity only if users have common items
            if np.sum(user_row > 0) > 0 and np.sum(other_row > 0) > 0:
                sim = cosine_similarity(user_row.reshape(1, -1), other_row.reshape(1, -1))[0][0]
                similarities.append((i, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    similar_users = similarities[:n_users]
    
    # Get recommendations
    recommendations = defaultdict(float)
    for similar_user, similarity in similar_users:
        similar_user_row = matrix[similar_user].toarray().flatten()
        for item_idx in range(len(similar_user_row)):
            if user_row[item_idx] == 0 and similar_user_row[item_idx] > 0:
                recommendations[item_idx] += similarity * similar_user_row[item_idx]
    
    # Sort recommendations
    recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    return recommendations

# Item-based collaborative filtering
def item_based_cf(matrix, user_idx, n_items=10, n_recommendations=10):
    # Get user's rated items
    user_row = matrix[user_idx].toarray().flatten()
    user_rated_items = np.where(user_row > 0)[0]
    
    # Get recommendations
    recommendations = defaultdict(float)
    for item_idx in user_rated_items:
        item_col = matrix.T[item_idx].toarray().flatten()
        
        # Find similar items
        for other_item_idx in range(matrix.shape[1]):
            if other_item_idx != item_idx and user_row[other_item_idx] == 0:
                other_item_col = matrix.T[other_item_idx].toarray().flatten()
                # Compute similarity only if items have common users
                if np.sum(item_col > 0) > 0 and np.sum(other_item_col > 0) > 0:
                    sim = cosine_similarity(item_col.reshape(1, -1), other_item_col.reshape(1, -1))[0][0]
                    recommendations[other_item_idx] += sim * user_row[item_idx]
    
    # Sort recommendations
    recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    return recommendations

# SVD-based collaborative filtering
def svd_based_cf(matrix, user_idx, n_factors=50, n_recommendations=10):
    # Perform SVD
    U, sigma, Vt = svds(matrix, k=n_factors)
    
    # Reconstruct the matrix
    sigma_diag = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U, sigma_diag), Vt)
    
    # Get the predicted ratings for user
    user_pred_ratings = predicted_ratings[user_idx]
    
    # Get actual ratings
    user_row = matrix[user_idx].toarray().flatten()
    
    # Create mask for unrated items
    unrated_items = np.where(user_row == 0)[0]
    
    # Get recommendations
    recommendations = [(item_idx, user_pred_ratings[item_idx]) 
                      for item_idx in unrated_items]
    
    # Sort by predicted rating
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n_recommendations]

# Evaluation metrics
def hit_ratio(recommended_items, test_items):
    hits = len(set(recommended_items) & set(test_items))
    return hits / len(test_items) if test_items else 0

def ndcg(recommended_items, test_items):
    dcg = 0
    idcg = 0
    
    for i, item in enumerate(recommended_items):
        if item in test_items:
            dcg += 1 / np.log2(i + 2)
    
    for i in range(min(len(test_items), len(recommended_items))):
        idcg += 1 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0

def mrr(recommended_items, test_items):
    for i, item in enumerate(recommended_items):
        if item in test_items:
            return 1 / (i + 1)
    return 0

# Evaluate recommendations
def evaluate(matrix, users, test_ratio=0.2, n_users=10, n_items=10, n_factors=50, n_recommendations=10):
    metrics = {
        'user_hr': [], 'user_ndcg': [], 'user_mrr': [],
        'item_hr': [], 'item_ndcg': [], 'item_mrr': [],
        'svd_hr': [], 'svd_ndcg': [], 'svd_mrr': []
    }
    
    # Sample users for evaluation
    sampled_users = random.sample(range(len(users)), min(50, len(users)))
    print(f"Evaluating recommendations for {len(sampled_users)} users...")
    
    for user_idx in tqdm(sampled_users, desc="Evaluating", ncols=80):
        # Get items user has rated
        user_row = matrix[user_idx].toarray().flatten()
        user_items = np.where(user_row > 0)[0]
        
        # Skip users with too few interactions
        if len(user_items) <= 2:
            continue
            
        # Select test items
        n_test = max(1, int(len(user_items) * test_ratio))
        test_items = random.sample(list(user_items), n_test)
        
        # Create training matrix
        train_matrix = matrix.copy()
        for item_idx in test_items:
            train_matrix[user_idx, item_idx] = 0
        
        # Get recommendations
        user_recs = user_based_cf(train_matrix, user_idx, n_users, n_recommendations)
        # item_recs = item_based_cf(train_matrix, user_idx, n_items, n_recommendations)
        svd_recs = svd_based_cf(train_matrix, user_idx, n_factors, n_recommendations)
        
        # Extract just the item indices
        user_rec_items = [item_idx for item_idx, _ in user_recs]
        svd_rec_items = [item_idx for item_idx, _ in svd_recs]
        
        # Calculate metrics
        metrics['user_hr'].append(hit_ratio(user_rec_items, test_items))
        metrics['user_ndcg'].append(ndcg(user_rec_items, test_items))
        metrics['user_mrr'].append(mrr(user_rec_items, test_items))
        
        metrics['svd_hr'].append(hit_ratio(svd_rec_items, test_items))
        metrics['svd_ndcg'].append(ndcg(svd_rec_items, test_items))
        metrics['svd_mrr'].append(mrr(svd_rec_items, test_items))
    
    # Calculate average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
    return avg_metrics

def print_recommendations(user_id, recommendations, original_items, movies=None, method_name=""):
    print(f"\n{method_name} recommendations for user {user_id}:")
    for item_idx, score in recommendations:
        item_id = original_items[item_idx]
        if movies is not None and item_id in movies['movieId'].values:
            movie_name = movies[movies['movieId'] == item_id]['title'].values[0]
            print(f"Movie ID: {item_id}, Score: {score:.2f}, Title: {movie_name}")
        else:
            print(f"Movie ID: {item_id}, Score: {score:.2f}")

def main():
    # Load and preprocess data
    ratings_df, movies = load_ml32m()

    # Use a sample of the data if it's too large
    #（may generate 16966441536 which will crush the program）
    # Adjust based on your hardware capabilities
    sample_size = 10000
    print(f"Using a sample of {sample_size} users for analysis...")
    
    matrix, users, items, user_mapper, item_mapper = preprocess_data(ratings_df, sample_size)
    
    # Example: Get recommendations for a specific user
    user_idx = 0  # First user in the sample
    original_user_id = users[user_idx]
    
    # Generate recommendations
    print(f"\nGenerating recommendations for user {original_user_id}...")
    
    user_recs = user_based_cf(matrix, user_idx)
    print_recommendations(original_user_id, user_recs, items, movies, "User-based")
    
    svd_recs = svd_based_cf(matrix, user_idx)
    print_recommendations(original_user_id, svd_recs, items, movies, "SVD-based")
    
    # Evaluate models
    print("\nEvaluating recommendation methods...")
    metrics = evaluate(matrix, users)
    
    print(f"\nEvaluation results:")
    print(f"User-based CF - HR@K: {metrics['user_hr']:.4f}, NDCG@K: {metrics['user_ndcg']:.4f}, MRR@K: {metrics['user_mrr']:.4f}")
    print(f"SVD-based CF - HR@K: {metrics['svd_hr']:.4f}, NDCG@K: {metrics['svd_ndcg']:.4f}, MRR@K: {metrics['svd_mrr']:.4f}")

if __name__ == "__main__":
    main()