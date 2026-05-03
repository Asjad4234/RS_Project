from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from Movie_Recommender_User_Input import (
    get_recommendations, find_best_match, popular_movies,
    get_hybrid_recommendations, user_similarity_df
)
from tmdb_client import get_full_movie_details

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
        if not data or 'user_id' not in data or 'title' not in data:
            return jsonify({'error': 'Please provide both user_id and title'}), 400

        user_id = data['user_id']
        title   = data['title']
        top_n        = data.get('top_n', 10)
        item_weight  = data.get('item_weight', 0.5)
        user_weight  = data.get('user_weight', 0.5)

        # Validate user_id is a valid integer in the dataset
        if user_id not in user_similarity_df.index:
            user_ids = sorted(user_similarity_df.index.tolist())
            return jsonify({
                'error': (
                    f"User ID {user_id} not found. "
                    f"Valid IDs are {user_ids[0]}–{user_ids[-1]}."
                )
            }), 404

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
                exact_match = find_best_match(movie_title, popular_movies.tolist())
                
                if exact_match:
                    recs = get_recommendations(exact_match, count=5)
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
