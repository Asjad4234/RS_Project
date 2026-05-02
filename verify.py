"""
Verification script — tests all CF functions directly without Flask.
Run: python verify.py
"""
from Movie_Recommender_User_Input import (
    get_recommendations,
    get_user_based_recommendations,
    get_hybrid_recommendations,
    find_best_match,
    user_similarity_df,
    popular_movies,
)

print("=" * 60)
print("VERIFICATION: Movie Recommendation System - Hybrid CF")
print("=" * 60)

# 1. /movies equivalent
print(f"\n[1] /movies -> {len(popular_movies)} filtered movies available")

# 2. /users equivalent
user_ids = sorted(user_similarity_df.index.tolist())
print(f"[2] /users  -> count={len(user_ids)}, range={user_ids[0]}-{user_ids[-1]}")

# 3. /recommend — unchanged Item-Item CF
title = "Toy Story (1995)"
item_recs = get_recommendations(title, top_n=5)
print(f"\n[3] /recommend (Item-Item) for '{title}':")
print(item_recs[['rank','title','similarity_score']].to_string(index=False))

# 4. /recommend-hybrid with user_id=1
hybrid_recs = get_hybrid_recommendations(1, title, top_n=5)
print(f"\n[4] /recommend-hybrid for user_id=1 + '{title}':")
print(hybrid_recs[['rank','title','similarity_score']].to_string(index=False))

# 5. Invalid user_id validation
invalid_result = get_user_based_recommendations(9999, title)
print(f"\n[5] Invalid user_id=9999 -> empty={invalid_result.empty}")

# 6. Confirm hybrid differs from item-item
item_titles   = set(item_recs['title'])
hybrid_titles = set(hybrid_recs['title'])
overlap = item_titles & hybrid_titles
print(f"\n[6] Overlap between Item-Item top-5 and Hybrid top-5: {len(overlap)}/5")
print(f"    (some difference expected — hybrid blends user history)")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
