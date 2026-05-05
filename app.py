from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from Movie_Recommender_User_Input import (
    get_recommendations, find_best_match, popular_movies,
    get_hybrid_recommendations, user_similarity_df,
    rating_counts, movies_with_genres, genre_names, filtered_df
)
from tmdb_client import get_full_movie_details

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions and return JSON."""
    # Pass through HTTP errors
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return jsonify({'error': str(e)}), e.code
    
    # Non-HTTP exceptions (crashes)
    import traceback
    error_msg = f"Unhandled Exception: {str(e)}\n{traceback.format_exc()}"
    print(error_msg)
    return jsonify({
        'error': str(e),
        'traceback': traceback.format_exc()
    }), 500

# Load poster cache at startup
POSTER_CACHE = {}
DETAILS_CACHE = {}

def load_poster_cache():
    """Load the precomputed poster cache from JSON file."""
    global POSTER_CACHE
    cache_file = os.path.join(os.path.dirname(__file__), 'movie_posters_cache.json')
    try:
        with open(cache_file, 'r') as f:
            POSTER_CACHE = json.load(f)
        print(f"✅ Loaded poster cache for {len(POSTER_CACHE)} movies")
    except FileNotFoundError:
        print(f"⚠️  Cache file not found: {cache_file}")
    except Exception as e:
        print(f"❌ Error loading cache: {e}")

def load_details_cache():
    """Load the comprehensive movie details cache from JSON file."""
    global DETAILS_CACHE
    cache_file = os.path.join(os.path.dirname(__file__), 'movie_details_cache.json')
    try:
        with open(cache_file, 'r') as f:
            DETAILS_CACHE = json.load(f)
        print(f"✅ Loaded details cache for {len(DETAILS_CACHE)} movies")
    except FileNotFoundError:
        print(f"⚠️  Details cache file not found: {cache_file}")
    except Exception as e:
        print(f"❌ Error loading details cache: {e}")

# Load caches when app starts
load_poster_cache()
load_details_cache()

# Helper function to enrich recommendations with poster URLs (from cache)
def enrich_recommendations_with_posters(recommendations):
    """Add poster_url to each recommendation from cache."""
    for rec in recommendations:
        if 'title' in rec:
            # Look up poster from cache
            poster_data = POSTER_CACHE.get(rec['title'], {})
            rec['poster_url'] = poster_data.get('poster_url')
    return recommendations

# ── GET /movies ───────────────────────────────────────────────────
@app.route('/movies', methods=['GET'])
def get_movies():
    try:
        movie_list = sorted(popular_movies.tolist())
        # Return movies with poster URLs from cache
        enriched_movies = []
        for title in movie_list:
            poster_data = POSTER_CACHE.get(title, {})
            enriched_movies.append({
                'title': title,
                'poster_url': poster_data.get('poster_url')
            })
        return jsonify(enriched_movies)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── GET /popular ─────────────────────────────────────────────────────
@app.route('/popular', methods=['GET'])
def get_popular():
    """Return the top 20 movies ranked by popularity_score = rating_count × avg_rating."""
    try:
        # Build popularity scores for every popular movie safely
        counts = rating_counts.reindex(popular_movies, fill_value=0)
        avg_ratings = filtered_df.groupby('title')['rating'].mean().reindex(popular_movies, fill_value=0)
        
        popularity = (counts * avg_ratings).sort_values(ascending=False).head(20)

        result = []
        for title in popularity.index:
            poster_data = POSTER_CACHE.get(title, {})
            result.append({
                'title': title,
                'poster_url': poster_data.get('poster_url'),
                'popularity_score': round(float(popularity[title]), 2)
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── POST /recommend-by-genres ────────────────────────────────────────
@app.route('/recommend-by-genres', methods=['POST'])
def recommend_by_genres():
    """
    Cold-start genre-based recommendations.
    Request: {"genres": ["Action", "Comedy"], "top_n": 20}
    Returns movies from popular_movies matching at least one genre,
    ranked by rating count descending.
    """
    try:
        data = request.get_json()
        genres = data.get('genres', [])
        top_n = min(data.get('top_n', 20), 50)

        if not genres:
            return jsonify({'error': 'Please provide at least one genre'}), 400

        # Validate genre names
        valid_genres = [g for g in genres if g in genre_names]
        if not valid_genres:
            return jsonify({'error': f'No valid genres. Choose from: {genre_names}'}), 400

        # Filter movies_with_genres to only popular movies
        popular_genre_df = movies_with_genres[movies_with_genres['title'].isin(popular_movies)].copy()

        # Match movies that have at least one of the requested genres
        genre_mask = popular_genre_df[valid_genres].max(axis=1) == 1
        matched = popular_genre_df[genre_mask].copy()

        if matched.empty:
            return jsonify({'genres': valid_genres, 'recommendations': []})

        # Rank by rating count
        # Rank by rating count
        matched['rating_count'] = matched['title'].map(rating_counts).fillna(0)
        matched = matched.sort_values('rating_count', ascending=False).head(top_n)

        result = []
        for _, row in matched.iterrows():
            poster_data = POSTER_CACHE.get(row['title'], {})
            result.append({
                'title': row['title'],
                'poster_url': poster_data.get('poster_url'),
                'rating_count': int(row['rating_count'])
            })

        return jsonify({
            'genres': valid_genres,
            'recommendations': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── GET /users ────────────────────────────────────────────────────
@app.route('/users', methods=['GET'])
def get_users():
    try:
        # Return the count and valid ID range for the frontend helper text
        user_ids = sorted(user_similarity_df.index.tolist())
        return jsonify({
            'count': len(user_ids),
            'range': f'{user_ids[0]}–{user_ids[-1]}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── GET /user-saved-movies/<user_id> ────────────────────────────────
@app.route('/user-saved-movies/<int:user_id>', methods=['GET'])
def get_user_saved_movies(user_id):
    """Return movies the user has rated highly (4 or 5) in the dataset."""
    try:
        from Movie_Recommender_User_Input import filtered_df
        
        # Filter for high ratings by this user
        user_high_ratings = filtered_df[
            (filtered_df['user_id'] == user_id) & 
            (filtered_df['rating'] >= 4)
        ].copy()
        
        # Sort by rating descending and then title
        user_high_ratings = user_high_ratings.sort_values(by=['rating'], ascending=False)
        
        result = []
        for title in user_high_ratings['title'].unique():
            poster_data = POSTER_CACHE.get(title, {})
            result.append({
                'title': title,
                'poster_url': poster_data.get('poster_url'),
                'rating': float(user_high_ratings[user_high_ratings['title'] == title]['rating'].iloc[0])
            })
            
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── GET /search ─────────────────────────────────────────────────────
@app.route('/search', methods=['GET'])
def search_movies_suggestions():
    """Return top 10 movies matching the query string for search suggestions."""
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    
    try:
        # Filter popular_movies list for titles containing the query
        matches = [title for title in popular_movies if query in title.lower()]
        # Sort matches by title length then alphabetically
        matches = sorted(matches, key=lambda x: (len(x), x))
        
        # Take top 10
        top_matches = matches[:10]
        
        result = []
        for title in top_matches:
            poster_data = POSTER_CACHE.get(title, {})
            result.append({
                'title': title,
                'poster_url': poster_data.get('poster_url')
            })
            
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── GET /movie-details/<title> ────────────────────────────────────
@app.route('/movie-details/<title>', methods=['GET'])
def get_movie_details(title):
    try:
        # First, try to get comprehensive details from cache
        if title in DETAILS_CACHE:
            cached_details = DETAILS_CACHE[title]
            # Filter out empty values
            if cached_details.get("overview") or cached_details.get("poster_url"):
                return jsonify(cached_details)
        
        # Try to fetch full details from TMDb (for movies not in cache)
        details = get_full_movie_details(title)
        
        # If TMDb search failed but we have poster cached, use minimal data
        if not details:
            poster_data = POSTER_CACHE.get(title, {})
            if poster_data.get('poster_url'):
                details = {
                    'title': title,
                    'poster_url': poster_data.get('poster_url'),
                    'overview': '',
                }
        
        # If still no data, return error
        if not details:
            return jsonify({'error': f"Movie '{title}' not found"}), 404
        
        return jsonify(details)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── POST /recommend ───────────────────────────────────────────────
# Existing endpoint — NOT modified
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        if not data or 'title' not in data:
            return jsonify({'error': 'Please provide a movie title'}), 400

        title = data['title']
        top_n = data.get('top_n', 10)

        matched_title = find_best_match(title)
        if not matched_title:
            return jsonify({'error': f"Movie '{title}' not found or has insufficient ratings"}), 404

        result_df = get_recommendations(matched_title, top_n=top_n)
        recommendations = result_df.to_dict(orient='records')
        
        # Enrich with poster URLs
        recommendations = enrich_recommendations_with_posters(recommendations)

        return jsonify({
            'movie': matched_title,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── POST /recommend-hybrid ────────────────────────────────────────
@app.route('/recommend-hybrid', methods=['POST'])
def recommend_hybrid():
    try:
        data = request.get_json()

        # Validate required fields
        user_id = data.get('user_id')
        title   = data.get('title')
        top_n        = data.get('top_n', 10)
        item_weight  = data.get('item_weight', 0.5)
        user_weight  = data.get('user_weight', 0.5)

        print(f"DEBUG: recommend-hybrid called with user_id={user_id}, title={title}")
        
        if user_id is None or title is None:
            return jsonify({'error': 'Please provide both user_id and title'}), 400

        # Validate user_id — COLD START MITIGATION
        # If user_id is not in our similarity matrix, fall back to item-only recommendations
        if user_id not in user_similarity_df.index:
            print(f"⚠️ User ID {user_id} not found. Falling back to item-only recommendations.")
            item_weight = 1.0
            user_weight = 0.0
        
        # Match the movie title (supports partial matching)
        matched_title = find_best_match(title)
        if not matched_title:
            return jsonify({'error': f"Movie '{title}' not found or has insufficient ratings"}), 404

        result_df = get_hybrid_recommendations(
            user_id, matched_title,
            top_n=top_n,
            item_weight=item_weight,
            user_weight=user_weight
        )
        
        recommendations = result_df.to_dict(orient='records')
        
        # Enrich with poster URLs
        recommendations = enrich_recommendations_with_posters(recommendations)

        return jsonify({
            'movie': matched_title,
            'user_id': user_id,
            'mode': 'hybrid',
            'item_weight': item_weight,
            'user_weight': user_weight,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── POST /recommend-from-mylist ──────────────────────────────────
@app.route('/recommend-from-mylist', methods=['POST'])
def recommend_from_mylist():
    """
    Generate recommendations based on user's "My List" (watchlist).
    Useful for personalized recommendations built from saved movies.
    
    Request body:
    {
        "mylist": ["Movie 1 (YYYY)", "Movie 2 (YYYY)", ...],
        "count": 10
    }
    """
    try:
        data = request.get_json()
        mylist = data.get('mylist', [])
        count = min(data.get('count', 10), 20)  # Max 20 recommendations
        
        if not mylist:
            return jsonify({'error': 'mylist cannot be empty'}), 400
        
        all_recommendations = {}
        
        # Get recommendations for each movie in the list
        for movie_title in mylist:
            try:
                # Find exact match in dataset
                exact_match = find_best_match(movie_title)
                
                if exact_match:
                    recs_df = get_recommendations(exact_match, top_n=5)
                    # Convert DataFrame to list of titles
                    recs = recs_df['title'].tolist()
                    for rec in recs:
                        if rec not in mylist:  # Don't recommend movies already in mylist
                            all_recommendations[rec] = all_recommendations.get(rec, 0) + 1
            except Exception as e:
                print(f"Error getting recommendations for {movie_title}: {e}")
                continue
        
        # Sort by frequency (movies that appear in multiple recommendations)
        sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        final_recs = [movie for movie, score in sorted_recs[:count]]
        
        # Enrich with posters
        enriched_recs = []
        for movie in final_recs:
            poster_data = POSTER_CACHE.get(movie, {})
            enriched_recs.append({
                'title': movie,
                'poster_url': poster_data.get('poster_url'),
                'similarity_score': all_recommendations.get(movie, 0) / len(mylist)
            })
        
        return jsonify({
            'mylist': mylist,
            'recommendations': enriched_recs
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
