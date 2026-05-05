from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import pandas as pd

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # python-dotenv not installed, use system env vars

from Movie_Recommender_User_Input import (
    get_recommendations, find_best_match, popular_movies,
    get_hybrid_recommendations, user_similarity_df,
    rating_counts, movies_with_genres, genre_names, filtered_df,
    build_content_similarity_matrix, get_content_based_recommendations,
    get_shared_genres
)
from tmdb_client import get_full_movie_details, get_popular_actors, get_movie_cast, get_animated_character_avatars

app = Flask(__name__)
# Enable CORS for the frontend URL from environment
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:8081")
CORS(app, origins=[FRONTEND_URL])

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

# ── Build Content-Based Similarity Matrix (Cold Start Fallback) ──
print("🔨 Building content-based similarity matrix...")
try:
    content_sim_df = build_content_similarity_matrix(movies_with_genres)
    print(f"✅ Content similarity matrix built for {len(content_sim_df)} movies")
except Exception as e:
    print(f"❌ Error building content similarity matrix: {e}")
    content_sim_df = None

# Helper function to enrich recommendations with poster URLs (from cache)
def enrich_recommendations_with_posters(recommendations):
    """Add poster_url to each recommendation from cache."""
    for rec in recommendations:
        if 'title' in rec:
            # Look up poster from cache
            poster_data = POSTER_CACHE.get(rec['title'], {})
            rec['poster_url'] = poster_data.get('poster_url')
    return recommendations


def generate_reason(
    movie_title,
    query_title,
    source,
    similarity_score=None,
    avg_rating=None,
    rating_count=None,
    matched_genres=None,
    neighbor_count=None
):
    """
    Generate a human-readable explanation for why a movie was recommended.
    
    Design principle: The reason should answer "why THIS movie?" not just
    "how did the algorithm work?" Users don't care about cosine similarity —
    they care about whether this movie is like one they already liked.
    """
    if source == 'content_based':
        if matched_genres:
            top_genres = matched_genres[:2]
            genres_str = ' & '.join(top_genres)
            return f"Matches the {genres_str} style of {query_title}"
        return f"Similar genre mix to {query_title}"

    if source == 'item_based' or source == 'collaborative':
        if similarity_score is not None:
            if similarity_score > 0.85:
                strength = "Very similar"
            elif similarity_score > 0.65:
                strength = "Similar"
            else:
                strength = "Somewhat similar"
            return f"{strength} viewing pattern to {query_title}"
        return f"Viewers of {query_title} also loved this"

    if source == 'user_based':
        if neighbor_count:
            return f"{neighbor_count} users with your taste rated this highly"
        return "Recommended by users with similar history"

    if source == 'hybrid_with_content':
        return f"Matches taste and genre profile of {query_title}"

    if source == 'popular':
        if avg_rating and rating_count:
            return f"Rated {avg_rating:.1f}/5 by {rating_count:,} viewers"
        return "Highly rated by the community"

    if source == 'genre_based':
        if matched_genres:
            return f"Matches your {matched_genres[0]} preference"
        return "Matches your genre preferences"

    if source == 'my_list':
        return f"Based on your watchlist"

    return "Recommended for you"


def build_recommendations_response(
    results_df,
    query_title,
    source,
    top_n=10
):
    """
    Convert a results DataFrame into the API response list.
    Each item includes: title, poster, overview, reason, similarity_score, source.
    Filters to only include movies with cached posters for clean demo.
    """
    output = []

    for _, row in results_df.iterrows():
        movie_title = row['title']

        # Look up cached metadata
        poster = POSTER_CACHE.get(movie_title, {}).get('poster_url', '')
        
        # Skip movies without cached posters (demo killer - blank grey cards)
        if not poster:
            continue
        overview = DETAILS_CACHE.get(movie_title, {}).get('overview', '')
        avg_rating = DETAILS_CACHE.get(movie_title, {}).get('vote_average')
        rating_count = DETAILS_CACHE.get(movie_title, {}).get('vote_count')

        # Compute matched genres (for content-based explainability)
        matched_genres = []
        if source in ('content_based', 'hybrid_with_content', 'genre_based'):
            matched_genres = get_shared_genres(query_title, movie_title, movies_with_genres)

        # Similarity score (whichever column is available)
        sim_score = (
            row.get('hybrid_score') or
            row.get('content_similarity_score') or
            row.get('final_score') or
            row.get('similarity_score')
        )

        reason = generate_reason(
            movie_title=movie_title,
            query_title=query_title,
            source=source,
            similarity_score=float(sim_score) if sim_score else None,
            avg_rating=avg_rating,
            rating_count=rating_count,
            matched_genres=matched_genres
        )

        output.append({
            'title': movie_title,
            'poster_url': poster,
            'overview': overview,
            'reason': reason,
            'similarity_score': round(float(sim_score), 4) if sim_score else None,
            'source': source
        })
        
        # Stop when we have top_n results with posters
        if len(output) >= top_n:
            break

    return output

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
            # Try content-based as fallback
            if content_sim_df is not None:
                print(f"⚠️ Movie '{title}' not found in CF matrix. Using content-based fallback.")
                cb_results = get_content_based_recommendations(title, content_sim_df, top_n=top_n)
                if not cb_results.empty:
                    recommendations = build_recommendations_response(
                        cb_results, title, 'content_based', top_n
                    )
                    return jsonify({
                        'movie': title,
                        'recommendations': recommendations
                    })
            
            return jsonify({'error': f"Movie '{title}' not found or has insufficient ratings"}), 404

        result_df = get_recommendations(matched_title, top_n=top_n)
        
        # Build response with reasons
        recommendations = build_recommendations_response(
            result_df, matched_title, 'item_based', top_n
        )

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

        # Match the movie title (supports partial matching)
        matched_title = find_best_match(title)
        
        # Validate user_id — COLD START MITIGATION
        # If user_id is not in our similarity matrix, fall back to item-only recommendations
        if user_id not in user_similarity_df.index:
            print(f"⚠️ User ID {user_id} not found. Falling back to item-only recommendations.")
            item_weight = 1.0
            user_weight = 0.0
        
        if not matched_title:
            # If matched_title is None, try content-based as fallback
            if content_sim_df is not None:
                print(f"⚠️ Movie '{title}' not found in CF matrix. Using content-based fallback.")
                cb_results = get_content_based_recommendations(title, content_sim_df, top_n=top_n)
                if not cb_results.empty:
                    recommendations = build_recommendations_response(
                        cb_results, title, 'content_based', top_n
                    )
                    return jsonify({
                        'movie': title,
                        'user_id': user_id,
                        'mode': 'content_based',
                        'recommendations': recommendations
                    })
            
            return jsonify({'error': f"Movie '{title}' not found or has insufficient ratings"}), 404

        # Step 1: Try collaborative filtering
        cf_results = get_hybrid_recommendations(
            user_id, matched_title,
            top_n=top_n * 2,  # Pull more for potential blending
            item_weight=item_weight,
            user_weight=user_weight
        )
        
        # Step 2: Check if CF returned enough results
        # If not (cold start for this movie), blend in content-based
        source = 'hybrid'
        if len(cf_results) < top_n // 2 and content_sim_df is not None:
            print(f"⚠️ Sparse CF results ({len(cf_results)} < {top_n // 2}). Blending with content-based.")
            cb_results = get_content_based_recommendations(
                matched_title, content_sim_df, top_n=top_n
            )
            
            if not cf_results.empty and not cb_results.empty:
                # Blend: normalise both score columns to [0,1] then weighted average
                cf_results['score_norm'] = cf_results['similarity_score'] / cf_results['similarity_score'].max()
                cb_results['score_norm'] = cb_results['content_similarity_score']
                
                merged = pd.merge(
                    cf_results[['title', 'score_norm']],
                    cb_results[['title', 'score_norm']],
                    on='title',
                    how='outer',
                    suffixes=('_cf', '_cb')
                )
                merged = merged.fillna(0)
                merged['final_score'] = 0.4 * merged['score_norm_cf'] + 0.6 * merged['score_norm_cb']
                cf_results = merged[['title', 'final_score']].sort_values('final_score', ascending=False)
                cf_results.columns = ['title', 'similarity_score']
                source = 'hybrid_with_content'
            elif not cb_results.empty:
                cf_results = cb_results.rename(columns={'content_similarity_score': 'similarity_score'})
                source = 'content_based'
        
        # Build response with reasons
        recommendations = build_recommendations_response(
            cf_results, matched_title, source, top_n
        )

        return jsonify({
            'movie': matched_title,
            'user_id': user_id,
            'mode': source,
            'item_weight': item_weight,
            'user_weight': user_weight,
            'recommendations': recommendations
        })

    except Exception as e:
        import traceback
        print(f"❌ Error in recommend-hybrid: {e}\n{traceback.format_exc()}")
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

# ── GET /avatars/popular ─────────────────────────────────────────
@app.route('/avatars/popular', methods=['GET'])
def get_avatars_popular():
    """
    Fetch popular actors with profile images to use as profile avatars.
    Great for user-profile selection.
    
    Returns:
        List of actors with profile images, names, and TMDB URLs
    """
    try:
        actors = get_popular_actors()
        if not actors:
            return jsonify({'error': 'No actors found', 'avatars': []}), 200
        
        return jsonify({'avatars': actors, 'count': len(actors)})
    except Exception as e:
        return jsonify({'error': str(e), 'avatars': []}), 200


# ── GET /avatars/all ────────────────────────────────────────────────
@app.route('/avatars/all', methods=['GET'])
def get_avatars_all():
    """
    Fetch all available avatars: popular actors + movie characters + animated characters.
    Returns a diverse set of avatars (50+).
    
    Returns:
        Combined list of actors, cast, and characters from various sources
    """
    try:
        all_avatars = []
        
        # Get popular actors
        popular = get_popular_actors()
        if popular:
            all_avatars.extend(popular[:10])
        
        # Get animated/character avatars from various movies
        animated = get_animated_character_avatars()
        if animated:
            all_avatars.extend(animated[:40])
        
        # Deduplicate by profile_url
        seen = set()
        unique_avatars = []
        for avatar in all_avatars:
            url = avatar.get('profile_url', '')
            if url and url not in seen:
                seen.add(url)
                unique_avatars.append(avatar)
        
        return jsonify({
            'avatars': unique_avatars,
            'count': len(unique_avatars),
            'total_found': len(unique_avatars)
        })
    except Exception as e:
        print(f"Error in /avatars/all: {e}")
        return jsonify({'error': str(e), 'avatars': []}), 200


# ── GET /avatars/movie/<movie_id>/cast ───────────────────────────
@app.route('/avatars/movie/<int:movie_id>/cast', methods=['GET'])
def get_avatars_cast(movie_id):
    """
    Fetch cast members from a specific movie with their profile images.
    Perfect for getting character-specific avatars.
    
    Args:
        movie_id: TMDB movie ID
    
    Returns:
        List of cast members with character names and profile images
    """
    try:
        cast = get_movie_cast(movie_id)
        if not cast:
            return jsonify({'error': f'No cast found for movie {movie_id}', 'avatars': []}), 200
        
        return jsonify({'avatars': cast, 'count': len(cast)})
    except Exception as e:
        return jsonify({'error': str(e), 'avatars': []}), 200


# ── GET /avatars/curated ─────────────────────────────────────────
@app.route('/avatars/curated', methods=['GET'])
def get_avatars_curated():
    """
    Get curated profile avatars based on iconic movie characters and actors.
    Returns a fixed set of movie cast members known to have good profile images.
    Enhanced to include more options and diverse characters.
    """
    try:
        # Movie IDs with great cast photos (including animated, classics, modern films)
        curated_movies = {
            278: "Shawshank Redemption",
            550: "Fight Club",
            603: "The Matrix",
            807: "Se7en",
            680: "Pulp Fiction",
            155: "The Dark Knight",
            13: "Forrest Gump",
            10193: "Toy Story 3",
            27: "Cinderella",
            120: "Fellowship of the Ring",
            10674: "Hercules",
            76: "Indiana Jones: Raiders",
        }
        
        all_avatars = []
        
        for movie_id, title in curated_movies.items():
            try:
                cast = get_movie_cast(movie_id)
                if cast:
                    # Add up to 4 cast members from each movie
                    for member in cast[:4]:
                        member['movie'] = title
                        member['category'] = 'cast'
                        all_avatars.append(member)
            except Exception:
                continue
        
        # Also add popular actors
        popular = get_popular_actors()
        if popular:
            for actor in popular[:5]:
                actor['category'] = 'actor'
                all_avatars.append(actor)
        
        # Deduplicate
        seen = set()
        unique = []
        for avatar in all_avatars:
            url = avatar.get('profile_url', '')
            if url and url not in seen:
                seen.add(url)
                unique.append(avatar)
        
        return jsonify({
            'avatars': unique[:60],  # Return up to 60 diverse avatars
            'count': len(unique[:60])
        })
    except Exception as e:
        print(f"Error in /avatars/curated: {e}")
        return jsonify({'error': str(e), 'avatars': []}), 200

if __name__ == '__main__':
    # Load configuration from environment
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    port = int(os.environ.get('FLASK_PORT', 5000))
    app.run(debug=debug, port=port)
