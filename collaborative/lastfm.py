import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random
from tqdm import tqdm
from scipy.sparse.linalg import svds

# Load the dataset
def load_lastfm(path="./datas/hetrec2011-lastfm-2k"):
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

# User-based collaborative filtering
# 根据与user_id的相似用户，推荐n个item（artist）给它
def user_based_cf(user_item_matrix_df, user_id, n_users=10, n_recommendations=10):
    # Calculate user similarity
    user_similarity = cosine_similarity(user_item_matrix_df)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix_df.index, columns=user_item_matrix_df.index)
    
    # Find similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:n_users+1].index
    
    # Get recommendations
    recommendations = defaultdict(float)

    # 对于当前没听过的艺术家，以相似用户喜欢的艺术家频次为准,
    # recommendations[item] = 累加（某user和其相似用户的相似度 * weight）
    for similar_user in similar_users:
        similarity_between_user = user_similarity_df.loc[user_id, similar_user]
        
        for item in user_item_matrix_df.columns:
            if user_item_matrix_df.loc[user_id, item] == 0 and user_item_matrix_df.loc[similar_user, item] > 0:
                recommendations[item] += similarity_between_user * user_item_matrix_df.loc[similar_user, item]
    
    # Sort recommendations
    recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    return recommendations

# Item-based collaborative filtering
def item_based_cf(user_item_matrix_df, user_id, n_items=10, n_recommendations=10):
    # 1. Calculate item similarity between items
    item_similarity = cosine_similarity(user_item_matrix_df.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix_df.columns, columns=user_item_matrix_df.columns)
    
    # 2. Get items the user has interacted with and weights > 0
    user_related_items = user_item_matrix_df.loc[user_id]
    related_items_id = user_related_items[user_related_items > 0].index.tolist()
    
    # Get recommendations
    recommendations = defaultdict(float)

    for item_id in related_items_id:
        item_weight = user_item_matrix_df.loc[user_id, item_id]
        
        # 2.1 Find n top items similar to current item, according to item_similarity_df
        similar_items = item_similarity_df[item_id].sort_values(ascending=False)[1:n_items+1]
        
        for similar_item, similarity_between_item in similar_items.items():
            # 2.2 only find the similar item is not in related_items_id, add it to recommendations
            if similar_item in related_items_id:
                continue
            # the below similar_item's weight must be 0
            recommendations[similar_item] += similarity_between_item * item_weight
    
    # Sort recommendations
    recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    return recommendations

# SVD-based collaborative filtering
def svd_based_cf(user_item_matrix_df, user_id, n_factors=50, n_recommendations=10):
    # Convert to numpy array for SVD
    user_item_array = user_item_matrix_df.values
    
    # Get user index
    user_idx = user_item_matrix_df.index.get_loc(user_id)
    
    # Ensure n_factors is not too large
    n_factors = min(n_factors, min(user_item_array.shape) - 1)
    
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
    candidate_items = np.argsort(-user_ratings * unrated_items)[:n_recommendations]
    
    # Convert back to original item IDs and predicted scores
    recommendations = [(user_item_matrix_df.columns[item_idx], user_ratings[item_idx]) 
                      for item_idx in candidate_items if unrated_items[item_idx]]
    
    return recommendations

# Evaluate recommendations
def evaluate(user_item_matrix, test_ratio=0.2, n_users=10, n_items=10, n_factors=50, n_recommendations=10):
    # Create a copy of the matrix to avoid modifying the original
    matrix = user_item_matrix.copy()
    
    # Metrics storage
    metrics = {
        'user_hr': [], 'user_ndcg': [], 'user_mrr': [],
        'item_hr': [], 'item_ndcg': [], 'item_mrr': [],
        'svd_hr': [], 'svd_ndcg': [], 'svd_mrr': []
    }
    # if not sample, cause 3 hours to execute evaluate but get similar results with using sample...
    sampled_users = random.sample(list(matrix.index), 50)
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
        
        # Get recommendations from all methods
        user_recs = user_based_cf(train_matrix, user_id, n_users, n_recommendations)
        item_recs = item_based_cf(train_matrix, user_id, n_items, n_recommendations)
        svd_recs = svd_based_cf(train_matrix, user_id, n_factors, n_recommendations)
        
        # Extract just the item IDs
        user_rec_items = [item_id for item_id, _ in user_recs]
        item_rec_items = [item_id for item_id, _ in item_recs]
        svd_rec_items = [item_id for item_id, _ in svd_recs]
        
        # Calculate metrics for user-based CF
        metrics['user_hr'].append(hit_ratio(user_rec_items, test_items))
        metrics['user_ndcg'].append(ndcg(user_rec_items, test_items))
        metrics['user_mrr'].append(mrr(user_rec_items, test_items))
        
        # Calculate metrics for item-based CF
        metrics['item_hr'].append(hit_ratio(item_rec_items, test_items))
        metrics['item_ndcg'].append(ndcg(item_rec_items, test_items))
        metrics['item_mrr'].append(mrr(item_rec_items, test_items))
        
        # Calculate metrics for SVD-based CF
        metrics['svd_hr'].append(hit_ratio(svd_rec_items, test_items))
        metrics['svd_ndcg'].append(ndcg(svd_rec_items, test_items))
        metrics['svd_mrr'].append(mrr(svd_rec_items, test_items))
    
    # Calculate average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
    
    return avg_metrics

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

def generate_recommendations(user_item_matrix_df, user_id, artists):
    # Get recommendations using different methods
    user_recommendations = user_based_cf(user_item_matrix_df, user_id)
    print_recommendations(user_id, user_recommendations, artists, "User-based")
    
    item_recommendations = item_based_cf(user_item_matrix_df, user_id)
    print_recommendations(user_id, item_recommendations, artists, "Item-based")
    
    svd_recommendations = svd_based_cf(user_item_matrix_df, user_id)
    print_recommendations(user_id, svd_recommendations, artists, "SVD-based")

def evaluate_models(user_item_matrix_df):
    print("\nEvaluating recommendation methods...")
    metrics = evaluate(user_item_matrix_df)
    print(f"\nEvaluation results:")
    print(f"User-based CF - HR@K: {metrics['user_hr']:.4f}, NDCG@K: {metrics['user_ndcg']:.4f}, MRR@K: {metrics['user_mrr']:.4f}")
    print(f"Item-based CF - HR@K: {metrics['item_hr']:.4f}, NDCG@K: {metrics['item_ndcg']:.4f}, MRR@K: {metrics['item_mrr']:.4f}")
    print(f"SVD-based CF - HR@K: {metrics['svd_hr']:.4f}, NDCG@K: {metrics['svd_ndcg']:.4f}, MRR@K: {metrics['svd_mrr']:.4f}")

def main():
    # Load and preprocess data
    user_artists, artists, user_item_matrix_df = load_and_preprocess_data()
    
    # Example: Get recommendations for a specific user
    user_id = user_item_matrix_df.index[0]  # First user in the dataset
    
    # Generate and print recommendations
    generate_recommendations(user_item_matrix_df, user_id, artists)
    
    # Evaluate the models
    evaluate_models(user_item_matrix_df)

if __name__ == "__main__":
    main()