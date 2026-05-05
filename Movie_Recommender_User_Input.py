# ============================================================================
# Project  : Movie Recommendation System
# Team     : Syed Asjad Ali Zaidi (22K-4234)
#            Affan Jan (22K-4475)
#            Muhammad Saad (22K-4407)
# Dataset  : MovieLens 100K (ml-100k) — GroupLens Research
# Technique: Item-Item Collaborative Filtering + Cosine Similarity
# ============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ── Configuration ─────────────────────────────────────────
MIN_RATINGS_THRESHOLD = 50   # Movies with fewer ratings are excluded
TOP_N = 10                   # Number of recommendations to return

# ── Data Loading ──────────────────────────────────────────
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('u.data', sep='\t', names=r_cols, encoding='latin-1')

# ── Load movies with genre one-hot columns ────────────────
genre_names = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

all_item_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_names
movies_full = pd.read_csv('u.item', sep='|', names=all_item_cols, encoding='latin-1')

# Keep a lightweight copy for the existing pipeline
movies = movies_full[['movie_id', 'title']].copy()

# Genre lookup table: movie_id → genre columns (exported for app.py)
movies_with_genres = movies_full[['movie_id', 'title'] + genre_names].copy()

# ── Data Preprocessing ───────────────────────────────────
# Drop columns that are not needed
ratings.drop('unix_timestamp', axis=1, inplace=True)

# Merge movies and ratings
merged_df = pd.merge(movies, ratings, on='movie_id')

# Handle missing values
merged_df.dropna(subset=['title', 'rating'], inplace=True)

# Apply minimum-ratings threshold filter
rating_counts = merged_df.groupby('title')['rating'].count()
popular_movies = rating_counts[rating_counts >= MIN_RATINGS_THRESHOLD].index
filtered_df = merged_df[merged_df['title'].isin(popular_movies)]

# ── Pivot Table Construction ─────────────────────────────
# Rows = movie titles, Columns = user_ids, Values = ratings
ratings_matrix = filtered_df.pivot_table(
    index='title', columns='user_id', values='rating'
)
ratings_matrix.fillna(0, inplace=True)

# ── Cosine Similarity Computation ────────────────────────
similarity_matrix = cosine_similarity(ratings_matrix.values)

# Store as a DataFrame with movie titles as index and columns
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=ratings_matrix.index,
    columns=ratings_matrix.index
)

# Zero out the diagonal to avoid recommending the same movie
np.fill_diagonal(similarity_df.values, 0)


def find_best_match(query):
    """
    Finds the best matching movie title in the database for a given query.
    Prioritizes exact matches, then finds all partial matches (case-insensitive)
    and returns the one with the most ratings.
    """
    if query in similarity_df.index:
        return query

    matches = [t for t in similarity_df.index if query.lower() in t.lower()]
    if not matches:
        return None

    # Return most popular match among partial matches
    return rating_counts[matches].idxmax()


# ── Recommendation Function ─────────────────────────────
def get_recommendations(movie_title, top_n=TOP_N):
    """
    Given a movie title, return the top-N most similar movies
    ranked by cosine similarity score (descending).

    Parameters
    ----------
    movie_title : str
        The title of the reference movie (must match dataset exactly).
    top_n : int
        Number of recommendations to return.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [rank, title, similarity_score]
    """
    # Find best match for the title
    target_title = find_best_match(movie_title)
    
    if not target_title:
        print(f"Sorry, '{movie_title}' is not in the database "
              f"(or has fewer than {MIN_RATINGS_THRESHOLD} ratings).")
        return pd.DataFrame(columns=['rank', 'title', 'similarity_score'])

    # Get similarity scores for the given movie, sort descending
    sim_scores = similarity_df[target_title].sort_values(ascending=False)
    top_similar = sim_scores.head(top_n)

    result = pd.DataFrame({
        'rank': range(1, len(top_similar) + 1),
        'title': top_similar.index,
        'similarity_score': top_similar.values
    })

    return result


# ══════════════════════════════════════════════════════════════════
# USER-BASED COLLABORATIVE FILTERING MODULE
# ══════════════════════════════════════════════════════════════════

# ── User Similarity Matrix ────────────────────────────────────────
# Transpose ratings_matrix so rows = users, columns = movies
user_ratings_matrix = ratings_matrix.T  # shape: (users, movies)

# Compute pairwise cosine similarity across all user vectors
user_similarity_matrix = cosine_similarity(user_ratings_matrix.values)

# Store as DataFrame with user_ids as both index and columns
user_similarity_df = pd.DataFrame(
    user_similarity_matrix,
    index=user_ratings_matrix.index,
    columns=user_ratings_matrix.index
)

# Zero out the diagonal so a user is not their own neighbour
np.fill_diagonal(user_similarity_df.values, 0)


def get_user_based_recommendations(user_id, movie_title, top_n=TOP_N):
    """
    Return top-N movie recommendations for a user based on what
    similar users have rated highly.

    Parameters
    ----------
    user_id : int
        Target user's ID (must exist in the filtered dataset).
    movie_title : str
        Reference movie title (used only for context; not filtered out here
        because the user may not have rated it).
    top_n : int
        Number of recommendations to return.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [rank, title, similarity_score]
    """
    # Validate that the user_id exists in our similarity matrix
    if user_id not in user_similarity_df.index:
        # COLD START FALLBACK: return the most popular movies ranked by rating count
        top_popular = rating_counts.loc[popular_movies].sort_values(ascending=False).head(top_n)
        result = pd.DataFrame({
            'rank': range(1, len(top_popular) + 1),
            'title': top_popular.index.tolist(),
            'similarity_score': [0.0] * len(top_popular),
            'cold_start': [True] * len(top_popular)
        })
        return result

    # Find the top-20 most similar users (neighbours)
    similar_users = (
        user_similarity_df[user_id]
        .sort_values(ascending=False)
        .head(20)
    )

    # Collect movies the target user has already rated — exclude these
    already_rated = set(
        filtered_df[filtered_df['user_id'] == user_id]['title'].unique()
    )

    # Build a weighted score for every candidate movie
    candidate_scores = {}

    for neighbour_id, similarity in similar_users.items():
        # Get all movies this neighbour rated
        neighbour_ratings = filtered_df[
            filtered_df['user_id'] == neighbour_id
        ][['title', 'rating']]

        for _, row in neighbour_ratings.iterrows():
            title = row['title']
            if title in already_rated:
                continue  # Skip movies the target user has seen

            if title not in candidate_scores:
                candidate_scores[title] = {'weighted_sum': 0.0, 'sim_sum': 0.0}

            # Weighted rating = similarity_weight × rating value
            candidate_scores[title]['weighted_sum'] += similarity * row['rating']
            candidate_scores[title]['sim_sum'] += abs(similarity)

    if not candidate_scores:
        return pd.DataFrame(columns=['rank', 'title', 'similarity_score'])

    # Compute normalised score for each candidate
    scores = {
        title: data['weighted_sum'] / data['sim_sum']
        for title, data in candidate_scores.items()
        if data['sim_sum'] > 0
    }

    # Sort and take top_n
    top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    if not top_movies:
        return pd.DataFrame(columns=['rank', 'title', 'similarity_score'])

    titles, raw_scores = zip(*top_movies)
    max_score = max(raw_scores)

    # Normalise scores to 0–1 range by dividing by the maximum score
    result = pd.DataFrame({
        'rank': range(1, len(titles) + 1),
        'title': list(titles),
        'similarity_score': [s / max_score for s in raw_scores]
    })

    return result


# ── Hybrid Scoring Function ───────────────────────────────────────
def get_hybrid_recommendations(user_id, movie_title, top_n=TOP_N,
                                item_weight=0.5, user_weight=0.5):
    """
    Blend Item-Item CF and User-Based CF scores into a single ranked list.

    Parameters
    ----------
    user_id : int
        Target user's ID.
    movie_title : str
        Reference movie title (may be partial — matched via find_best_match).
    top_n : int
        Number of final recommendations to return.
    item_weight : float
        Weight given to item-item similarity scores (default 0.5).
    user_weight : float
        Weight given to user-based scores (default 0.5).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [rank, title, similarity_score]
    """
    # COLD START: If user is unknown, force item-only mode
    if user_id not in user_similarity_df.index:
        item_weight = 1.0
        user_weight = 0.0

    # Normalise weights so they always sum to 1.0
    total = item_weight + user_weight
    if total == 0:
        item_weight, user_weight = 0.5, 0.5
    else:
        item_weight /= total
        user_weight /= total

    # Pull a wider pool (50) from each CF method for better blending
    item_df = get_recommendations(movie_title, top_n=50)
    user_df = get_user_based_recommendations(user_id, movie_title, top_n=50)

    # Rename columns before merging to distinguish the two score sources
    if not item_df.empty:
        item_df = item_df[['title', 'similarity_score']].rename(
            columns={'similarity_score': 'item_similarity_score'}
        )
    else:
        item_df = pd.DataFrame(columns=['title', 'item_similarity_score'])

    if not user_df.empty:
        user_df = user_df[['title', 'similarity_score']].rename(
            columns={'similarity_score': 'user_similarity_score'}
        )
    else:
        user_df = pd.DataFrame(columns=['title', 'user_similarity_score'])

    # Outer join on title — missing scores from either side become 0
    merged = pd.merge(item_df, user_df, on='title', how='outer').fillna(0)

    if merged.empty:
        return pd.DataFrame(columns=['rank', 'title', 'similarity_score'])

    # Compute the blended hybrid score
    merged['similarity_score'] = (
        item_weight * merged['item_similarity_score'] +
        user_weight * merged['user_similarity_score']
    )

    # Sort by hybrid score descending and take top_n
    merged = (
        merged.sort_values('similarity_score', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    result = pd.DataFrame({
        'rank': range(1, len(merged) + 1),
        'title': merged['title'].values,
        'similarity_score': merged['similarity_score'].values
    })

    return result


# ══════════════════════════════════════════════════════════════════
# CONTENT-BASED FILTERING MODULE (Cold Start Fallback)
# ══════════════════════════════════════════════════════════════════

def build_content_similarity_matrix(movies_df=None):
    """
    Compute pairwise cosine similarity on the binary genre matrix.
    
    Why cosine similarity? It measures the angle between genre vectors,
    so a pure Action movie and a mixed Action/Drama share partial similarity
    rather than being fully penalised for not matching exactly.
    
    Args:
        movies_df: DataFrame with title and genre columns. If None, uses movies_with_genres.
    
    Returns:
        DataFrame: rows and columns are movie titles, values are cosine similarities [0,1]
    """
    if movies_df is None:
        movies_df = movies_with_genres.copy()
    
    # Extract genre columns for all movies
    genre_matrix = movies_df.set_index('title')[genre_names].values
    similarity_matrix = cosine_similarity(genre_matrix)
    
    return pd.DataFrame(
        similarity_matrix,
        index=movies_df['title'],
        columns=movies_df['title']
    )


def get_content_based_recommendations(
    query_title,
    content_sim_df,
    top_n=10,
    exclude_titles=None
):
    """
    Return top-N movies most similar to query_title by genre vector.
    
    This is the PRIMARY fallback for:
      - Movies with fewer than MIN_RATINGS_THRESHOLD ratings
      - Any movie not found in the collaborative filtering matrix
      - New movies added after the ratings dataset was built
    
    Args:
        query_title: The movie to base recommendations on
        content_sim_df: Precomputed cosine similarity DataFrame (from build_content_similarity_matrix)
        top_n: How many results to return
        exclude_titles: Titles to remove from results (e.g., already in My List)
    
    Returns:
        DataFrame with columns: [rank, title, content_similarity_score]
    """
    if query_title not in content_sim_df.index:
        # Try partial match (handles minor title formatting differences)
        matches = [t for t in content_sim_df.index if query_title.lower() in t.lower()]
        if not matches:
            return pd.DataFrame(columns=['rank', 'title', 'content_similarity_score'])
        query_title = matches[0]
    
    scores = content_sim_df[query_title].drop(query_title)  # Exclude self
    
    if exclude_titles:
        scores = scores.drop(
            labels=[t for t in exclude_titles if t in scores.index],
            errors='ignore'
        )
    
    top = scores.nlargest(top_n).reset_index()
    top.columns = ['title', 'content_similarity_score']
    top['rank'] = range(1, len(top) + 1)
    
    return top[['rank', 'title', 'content_similarity_score']]


def get_shared_genres(title_a, title_b, movies_df=None):
    """
    Return genre labels shared between two movies.
    Used to produce natural-language genre overlap explanations.
    
    Returns:
        List of genre strings (e.g., ['Action', 'Drama'])
    """
    if movies_df is None:
        movies_df = movies_with_genres
    
    try:
        row_a = movies_df[movies_df['title'] == title_a].iloc[0]
        row_b = movies_df[movies_df['title'] == title_b].iloc[0]
        shared = [g for g in genre_names if row_a[g] == 1 and row_b[g] == 1]
        return shared
    except (IndexError, KeyError):
        return []


# ── Main ─────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        user_inp = input(
            'Enter the reference movie title based on which '
            'recommendations are to be made: '
        )
        # Find the best match first so we can display it
        matched_title = find_best_match(user_inp)
        recommendations = get_recommendations(user_inp)
        
        if not recommendations.empty:
            print(f"\nRecommended movies based on your choice of "
                  f"'{matched_title}':\n")
            print(recommendations.to_string(index=False))
    except Exception as e:
        print(f"An error occurred: {e}")
