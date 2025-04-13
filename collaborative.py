import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random
from tqdm import tqdm

# Load the dataset
def load_dataset(path="./datas/hetrec2011-lastfm-2k"):
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

# Evaluate recommendations
def evaluate(user_item_matrix, test_ratio=0.2, n_users=10, n_items=10, n_recommendations=10):
    # Create a copy of the matrix to avoid modifying the original
    matrix = user_item_matrix.copy()
    
    # Metrics storage
    metrics = {
        'user_hr': [], 'user_ndcg': [], 'user_mrr': [],
        'item_hr': [], 'item_ndcg': [], 'item_mrr': []
    }
    
    # Add progress bar
    total_users = len(matrix.index)
    print(f"Evaluating recommendations for {total_users} users...")
    
    # For each user, hide some interactions as test data
    for user_id in tqdm(matrix.index, desc="Evaluating", ncols=80):
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
        
        # Get recommendations from both methods
        user_recs = user_based_cf(train_matrix, user_id, n_users, n_recommendations)
        item_recs = item_based_cf(train_matrix, user_id, n_items, n_recommendations)
        
        # Extract just the item IDs
        user_rec_items = [item_id for item_id, _ in user_recs]
        item_rec_items = [item_id for item_id, _ in item_recs]
        
        # Calculate metrics for user-based CF
        metrics['user_hr'].append(hit_ratio(user_rec_items, test_items))
        metrics['user_ndcg'].append(ndcg(user_rec_items, test_items))
        metrics['user_mrr'].append(mrr(user_rec_items, test_items))
        
        # Calculate metrics for item-based CF
        metrics['item_hr'].append(hit_ratio(item_rec_items, test_items))
        metrics['item_ndcg'].append(ndcg(item_rec_items, test_items))
        metrics['item_mrr'].append(mrr(item_rec_items, test_items))
        print(metrics)
    
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

# Main function
def main():
    # Load and preprocess data
    user_artists, artists = load_dataset()
    user_item_matrix_df = preprocess_data(user_artists)

    # Example: Get recommendations for a specific user
    user_id = user_item_matrix_df.index[0]  # First user in the dataset
    # user_id = int(input("input the user id:"))
    
    print(f"User-based recommendations for user {user_id}:")
    user_recommendations = user_based_cf(user_item_matrix_df, user_id)
    for item_id, score in user_recommendations:
        artist_name = artists[artists['id'] == item_id]['name'].values[0] if item_id in artists['id'].values else "Unknown"
        print(f"Artist ID: {item_id}, Score: {score:.2f}, Name: {artist_name}")
    
    print(f"\nItem-based recommendations for user {user_id}:")
    item_recommendations = item_based_cf(user_item_matrix_df, user_id)
    for item_id, score in item_recommendations:
        artist_name = artists[artists['id'] == item_id]['name'].values[0] if item_id in artists['id'].values else "Unknown"
        print(f"Artist ID: {item_id}, Score: {score:.2f}, Name: {artist_name}")
    
    # Evaluate the models
    metrics = evaluate(user_item_matrix_df)
    print(f"\nEvaluation results:")
    print(f"User-based CF - HR@K: {metrics['user_hr']:.4f}, NDCG@K: {metrics['user_ndcg']:.4f}, MRR@K: {metrics['user_mrr']:.4f}")
    print(f"Item-based CF - HR@K: {metrics['item_hr']:.4f}, NDCG@K: {metrics['item_ndcg']:.4f}, MRR@K: {metrics['item_mrr']:.4f}")

if __name__ == "__main__":
    main()