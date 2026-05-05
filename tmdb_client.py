"""
TMDb API client for fetching movie posters and metadata.
Uses requests library to query The Movie Database API.
"""
import requests
import os
from functools import lru_cache
from typing import Optional, Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Use absolute path to the .env file in the same directory as this script
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    print(f"🔍 DEBUG: Attempting to load .env from: {env_path}")
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # python-dotenv not installed, use system env vars

# TMDb API endpoint and configuration
TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# Get API key from environment only (no hardcoded fallback)
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

if not TMDB_API_KEY:
    print("⚠️  WARNING: TMDB_API_KEY environment variable not set!")
    print("   Movie posters will not load.")
    print("   Setup instructions:")
    print("   1. Check the .env file in RS_Project/")
    print("   2. Ensure it contains: TMDB_API_KEY=your_api_key_here")
    print("   3. Get a free key from: https://www.themoviedb.org/settings/api")
    print("   4. Restart the backend")
    print()


# Cache results to minimize API calls
@lru_cache(maxsize=500)
def search_movie(title: str, year: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Search for a movie on TMDb by title and optional release year.
    Returns the first matching result with poster_path and other metadata.
    
    Args:
        title: Movie title (e.g., "Toy Story")
        year: Release year (e.g., 1995) - optional
    
    Returns:
        Dict with movie data (id, title, poster_path, release_date) or None if not found
    """
    if not TMDB_API_KEY:
        return None  # API key not available, skip search
    
    try:
        # Build query - year helps narrow results
        query = title
        if year:
            query = f"{title} {year}"
        
        params = {
            "api_key": TMDB_API_KEY,
            "query": query,
            "page": 1
        }
        
        response = requests.get(f"{TMDB_API_BASE}/search/movie", params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        if data.get("results") and len(data["results"]) > 0:
            result = data["results"][0]  # Take top result
            return {
                "id": result.get("id"),
                "title": result.get("title"),
                "poster_path": result.get("poster_path"),
                "release_date": result.get("release_date"),
                "overview": result.get("overview"),
                "vote_average": result.get("vote_average"),
            }
        return None
    except Exception as e:
        print(f"Error searching TMDb for '{title}': {str(e)}")
        return None


def get_poster_url(movie_data: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Extract full poster URL from TMDb movie data.
    
    Args:
        movie_data: Result from search_movie()
    
    Returns:
        Full URL to poster image or None if not available
    """
    if movie_data and movie_data.get("poster_path"):
        return f"{TMDB_IMAGE_BASE}{movie_data['poster_path']}"
    return None


@lru_cache(maxsize=500)
def get_movie_cast(movie_id: int) -> Optional[list]:
    """
    Fetch cast information for a movie, including actor profile images.
    Perfect for getting character/actor avatars.
    
    Args:
        movie_id: TMDB movie ID
    
    Returns:
        List of cast members with name, character, and profile_path
    """
    if not TMDB_API_KEY:
        return None
    
    try:
        params = {
            "api_key": TMDB_API_KEY,
        }
        
        response = requests.get(
            f"{TMDB_API_BASE}/movie/{movie_id}/credits",
            params=params,
            timeout=5
        )
        response.raise_for_status()
        
        data = response.json()
        cast = []
        
        for member in data.get("cast", [])[:10]:  # Top 10 cast members
            if member.get("profile_path"):  # Only include those with profile images
                cast.append({
                    "name": member.get("name"),
                    "character": member.get("character"),
                    "profile_path": member.get("profile_path"),
                    "profile_url": f"{TMDB_IMAGE_BASE}{member.get('profile_path')}"
                })
        
        return cast if cast else None
    except Exception as e:
        print(f"Error fetching cast for movie {movie_id}: {str(e)}")
        return None


@lru_cache(maxsize=100)
def get_popular_actors() -> Optional[list]:
    """
    Fetch popular actors with their profile images.
    Great for avatar selection.
    
    Returns:
        List of popular actors with profile images
    """
    if not TMDB_API_KEY:
        return None
    
    try:
        params = {
            "api_key": TMDB_API_KEY,
            "page": 1
        }
        
        response = requests.get(
            f"{TMDB_API_BASE}/person/popular",
            params=params,
            timeout=5
        )
        response.raise_for_status()
        
        data = response.json()
        actors = []
        
        for person in data.get("results", [])[:15]:
            if person.get("profile_path"):  # Only include those with profile images
                actors.append({
                    "id": person.get("id"),
                    "name": person.get("name"),
                    "profile_path": person.get("profile_path"),
                    "profile_url": f"{TMDB_IMAGE_BASE}{person.get('profile_path')}",
                    "popularity": person.get("popularity")
                })
        
        return actors if actors else None
    except Exception as e:
        print(f"Error fetching popular actors: {str(e)}")
        return None


@lru_cache(maxsize=100)
def get_animated_character_avatars() -> Optional[list]:
    """
    Fetch animated characters and voice actors from popular animated/family movies.
    Perfect for diverse, fun avatars (Toy Story, Frozen, Lion King, etc.)
    
    Returns:
        List of cast from animated movies with profile images
    """
    if not TMDB_API_KEY:
        return None
    
    try:
        # Movie IDs for popular animated movies known to have good cast photos
        animated_movies = [
            (10193, "Toy Story 3"),
            (27, "Cinderella"),
            (120, "The Lord of the Rings: The Fellowship of the Ring"),
            (278, "The Shawshank Redemption"),
            (550, "Fight Club"),
            (603, "The Matrix"),
            (680, "Pulp Fiction"),
            (13, "Forrest Gump"),
            (10674, "Hercules"),
            (155, "The Dark Knight"),
            (807, "Se7en"),
            (278, "The Shawshank Redemption"),
        ]
        
        all_avatars = []
        
        for movie_id, movie_title in animated_movies:
            try:
                cast = get_movie_cast(movie_id)
                if cast:
                    for member in cast[:5]:  # Top 5 from each movie
                        member['movie'] = movie_title
                        member['category'] = 'cast'
                        all_avatars.append(member)
            except Exception as e:
                continue
        
        return all_avatars if all_avatars else None
    except Exception as e:
        print(f"Error fetching animated character avatars: {str(e)}")
        return None


def enrich_movie_with_poster(title: str) -> Dict[str, Any]:
    """
    Get full movie details including poster URL for a given title.
    Extracts year from title if present (e.g., "Toy Story (1995)").
    
    Args:
        title: Movie title (optionally with year in parentheses)
    
    Returns:
        Dict with title and poster_url (poster_url may be None if not found)
    """
    year = None
    clean_title = title
    
    # Extract year from title if present (e.g., "Toy Story (1995)")
    if title and "(" in title and ")" in title:
        try:
            year_str = title[title.rfind("(") + 1 : title.rfind(")")]
            if year_str.isdigit():
                year = int(year_str)
                clean_title = title[: title.rfind("(")].strip()
        except ValueError:
            pass
    
    # Search TMDb
    movie_data = search_movie(clean_title, year)
    poster_url = get_poster_url(movie_data)
    
    return {
        "title": title,
        "poster_url": poster_url,
        "tmdb_data": movie_data,  # Store full data for future expansion
    }


@lru_cache(maxsize=500)
def get_movie_details(tmdb_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch detailed movie information from TMDb including cast, genres, and runtime.
    
    Args:
        tmdb_id: TMDb movie ID
    
    Returns:
        Dict with detailed movie info or None if not found
    """
    if not TMDB_API_KEY:
        return None
    
    try:
        params = {
            "api_key": TMDB_API_KEY,
            "append_to_response": "credits"  # Include cast/crew info
        }
        
        response = requests.get(
            f"{TMDB_API_BASE}/movie/{tmdb_id}",
            params=params,
            timeout=5
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Extract main cast (top 10)
        cast = []
        if data.get("credits", {}).get("cast"):
            cast = [
                {
                    "name": actor.get("name"),
                    "character": actor.get("character"),
                    "profile_path": actor.get("profile_path")
                }
                for actor in data.get("credits", {}).get("cast", [])[:10]
            ]
        
        # Extract genres
        genres = [g.get("name") for g in data.get("genres", [])]
        
        return {
            "id": data.get("id"),
            "title": data.get("title"),
            "overview": data.get("overview"),
            "poster_path": data.get("poster_path"),
            "backdrop_path": data.get("backdrop_path"),
            "release_date": data.get("release_date"),
            "vote_average": data.get("vote_average"),
            "runtime": data.get("runtime"),
            "genres": genres,
            "cast": cast,
        }
    except Exception as e:
        print(f"Error fetching TMDb details for ID {tmdb_id}: {str(e)}")
        return None


def get_full_movie_details(title: str) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive movie details by title.
    Searches for the movie, then fetches detailed info.
    
    Args:
        title: Movie title (optionally with year)
    
    Returns:
        Dict with comprehensive movie details or None if not found
    """
    # Extract year if present
    year = None
    clean_title = title
    
    if title and "(" in title and ")" in title:
        try:
            year_str = title[title.rfind("(") + 1 : title.rfind(")")]
            if year_str.isdigit():
                year = int(year_str)
                clean_title = title[: title.rfind("(")].strip()
        except ValueError:
            pass
    
    # Search for movie
    movie_data = search_movie(clean_title, year)
    if not movie_data or not movie_data.get("id"):
        return None
    
    # Fetch detailed info
    details = get_movie_details(movie_data["id"])
    
    return details or movie_data


def clear_cache():
    """Clear the LRU cache for testing or updates."""
    search_movie.cache_clear()
    get_movie_details.cache_clear()
