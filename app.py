from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from Movie_Recommender_User_Input import (
    get_recommendations, find_best_match, popular_movies,
    get_hybrid_recommendations, user_similarity_df
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load poster cache at startup
POSTER_CACHE = {}

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
        print("   Posters will not be displayed. Run build_poster_cache_fast.py first.")
    except Exception as e:
        print(f"❌ Error loading cache: {e}")

# Load cache when app starts
load_poster_cache()

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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
