import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random

# Load the dataset
def load_dataset(path="./hetrec2011-lastfm-2k"):
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
def evaluate_hr_k(user_item_matrix, test_ratio=0.2, n_users=10, n_items=10, n_recommendations=10):
    # Split data into train and test
    train_matrix = user_item_matrix.copy()
    test_set = []
    
    for user_id in user_item_matrix.index:
        user_items = user_item_matrix.loc[user_id]
        user_listed_items_id = user_items[user_items > 0].index.tolist()
        
        if len(user_listed_items_id) > 5:  # Only test users with enough interactions
            # Sample items for testing
            test_items = np.random.choice(user_listed_items_id, size=int(len(user_listed_items_id) * test_ratio), replace=False)
            for item in test_items:
                test_set.append((user_id, item, user_item_matrix.loc[user_id, item]))
                train_matrix.loc[user_id, item] = 0

    test_set = random.sample(test_set, 1)

    # Evaluate user-based CF
    user_based_precision = []
    for user_id, item_id, _ in test_set:
        recommendations = user_based_cf(train_matrix, user_id, n_users, n_recommendations)
        recommended_items = [item for item, score in recommendations]
        if item_id in recommended_items:
            user_based_precision.append(1)
        else:
            user_based_precision.append(0)
    
    # Evaluate item-based CF
    item_based_precision = []
    for user_id, item_id, _ in test_set:
        recommendations = item_based_cf(train_matrix, user_id, n_items, n_recommendations)
        recommended_items = [item for item, _ in recommendations]
        if item_id in recommended_items:
            item_based_precision.append(1)
        else:
            item_based_precision.append(0)
    
    return np.mean(user_based_precision), np.mean(item_based_precision)

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
    user_precision, item_precision = evaluate_hr_k(user_item_matrix_df)
    print(f"\nEvaluation results:")
    print(f"User-based CF precision: {user_precision:.4f}")
    print(f"Item-based CF precision: {item_precision:.4f}")

if __name__ == "__main__":
    main()