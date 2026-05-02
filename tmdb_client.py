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
    load_dotenv()
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
    print("   1. Create a .env file in RS_Back/")
    print("   2. Add: TMDB_API_KEY=your_api_key_here")
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


def clear_cache():
    """Clear the LRU cache for testing or updates."""
    search_movie.cache_clear()
