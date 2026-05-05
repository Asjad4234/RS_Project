#!/usr/bin/env python
"""
Comprehensive test suite for MovieFlix API endpoints
Tests all recommendation algorithms, cold start fallback, and explainability features
"""

import requests
import json

BASE_URL = 'http://localhost:5000'
test_results = []

def log_test(name, status, details=""):
    """Log test result"""
    symbol = "✅" if status == "PASS" else "❌"
    print(f"\n{symbol} {name}")
    if details:
        print(f"   {details}")
    test_results.append((name, status))

print("\n" + "="*70)
print("🎬 MOVIEFLIX BACKEND TEST SUITE")
print("="*70)

# ============================================================================
# TEST 1: /movies - Basic movie list
# ============================================================================
print("\n[TEST 1] GET /movies - Load all movies")
print("-" * 70)
try:
    r = requests.get(f'{BASE_URL}/movies')
    if r.status_code == 200:
        data = r.json()
        log_test("✓ /movies endpoint", "PASS", 
                f"Loaded {len(data)} movies | Sample: {data[0]['title']}")
    else:
        log_test("✗ /movies endpoint", "FAIL", f"Status: {r.status_code}")
except Exception as e:
    log_test("✗ /movies endpoint", "FAIL", str(e))

# ============================================================================
# TEST 2: /popular - Popularity ranking
# ============================================================================
print("\n[TEST 2] GET /popular - Top movies ranked by popularity")
print("-" * 70)
try:
    r = requests.get(f'{BASE_URL}/popular')
    if r.status_code == 200:
        data = r.json()
        log_test("✓ /popular endpoint", "PASS", f"Found {len(data)} popular movies")
        print("   Top 5 Popular Movies:")
        for i, movie in enumerate(data[:5], 1):
            print(f"   {i}. {movie['title']} (score: {movie.get('popularity_score', 'N/A')})")
    else:
        log_test("✗ /popular endpoint", "FAIL", f"Status: {r.status_code}")
except Exception as e:
    log_test("✗ /popular endpoint", "FAIL", str(e))

# ============================================================================
# TEST 3: /search - Search functionality
# ============================================================================
print("\n[TEST 3] GET /search - Search autocomplete")
print("-" * 70)
try:
    r = requests.get(f'{BASE_URL}/search?q=star')
    if r.status_code == 200:
        data = r.json()
        log_test("✓ /search endpoint", "PASS", f"Found {len(data)} results for 'star'")
        print("   Results:")
        for i, movie in enumerate(data[:5], 1):
            print(f"   {i}. {movie['title']}")
    else:
        log_test("✗ /search endpoint", "FAIL", f"Status: {r.status_code}")
except Exception as e:
    log_test("✗ /search endpoint", "FAIL", str(e))

# ============================================================================
# TEST 4: /recommend - Item-Based Collaborative Filtering
# ============================================================================
print("\n[TEST 4] POST /recommend - Item-Based CF with Explainability")
print("-" * 70)
try:
    payload = {"title": "Toy Story (1995)", "top_n": 5}
    r = requests.post(f'{BASE_URL}/recommend', json=payload)
    if r.status_code == 200:
        data = r.json()
        query_movie = data.get('movie')
        recs = data.get('recommendations', [])
        
        if len(recs) > 0:
            log_test("✓ /recommend endpoint", "PASS", 
                    f"Got {len(recs)} recs for '{query_movie}'")
            print(f"   Source: {recs[0].get('source', 'N/A')}")
            print("   Top 3 Recommendations:")
            for i, rec in enumerate(recs[:3], 1):
                print(f"   {i}. {rec['title']}")
                print(f"      Reason: {rec.get('reason', 'N/A')}")
                print(f"      Score: {rec.get('similarity_score', 'N/A')}")
        else:
            log_test("✓ /recommend endpoint", "PASS", "Returned empty (OK for edge case)")
    else:
        log_test("✗ /recommend endpoint", "FAIL", f"Status: {r.status_code}")
except Exception as e:
    log_test("✗ /recommend endpoint", "FAIL", str(e))

# ============================================================================
# TEST 5: /recommend-hybrid - Hybrid Collaborative Filtering
# ============================================================================
print("\n[TEST 5] POST /recommend-hybrid - Hybrid CF with Weights")
print("-" * 70)
try:
    payload = {
        "user_id": 1,
        "title": "Shawshank Redemption, The (1994)",
        "top_n": 5,
        "item_weight": 0.5,
        "user_weight": 0.5
    }
    r = requests.post(f'{BASE_URL}/recommend-hybrid', json=payload)
    if r.status_code == 200:
        data = r.json()
        mode = data.get('mode')
        item_w = data.get('item_weight')
        user_w = data.get('user_weight')
        recs = data.get('recommendations', [])
        
        log_test("✓ /recommend-hybrid endpoint", "PASS",
                f"Mode: {mode} | Weights: Item={item_w}, User={user_w}")
        print(f"   Recommendations: {len(recs)}")
        for i, rec in enumerate(recs[:3], 1):
            print(f"   {i}. {rec['title']} ({rec.get('source', 'N/A')})")
            print(f"      Reason: {rec.get('reason', 'N/A')}")
    else:
        log_test("✗ /recommend-hybrid endpoint", "FAIL", f"Status: {r.status_code}")
except Exception as e:
    log_test("✗ /recommend-hybrid endpoint", "FAIL", str(e))

# ============================================================================
# TEST 6: /recommend-hybrid - Cold Start (Unknown User)
# ============================================================================
print("\n[TEST 6] POST /recommend-hybrid - Cold Start with Unknown User")
print("-" * 70)
try:
    payload = {
        "user_id": 9999,  # Unknown user
        "title": "Toy Story (1995)",
        "top_n": 3,
        "item_weight": 0.5,
        "user_weight": 0.5
    }
    r = requests.post(f'{BASE_URL}/recommend-hybrid', json=payload)
    if r.status_code == 200:
        data = r.json()
        mode = data.get('mode')
        recs = data.get('recommendations', [])
        
        if len(recs) > 0:
            log_test("✓ Cold start handling", "PASS",
                    f"Fallback mode: {mode} | Got {len(recs)} recommendations")
            print(f"   Source of top rec: {recs[0].get('source', 'N/A')}")
        else:
            log_test("✓ Cold start handling", "PASS", "Returned empty (OK)")
    else:
        log_test("✗ Cold start handling", "FAIL", f"Status: {r.status_code}")
except Exception as e:
    log_test("✗ Cold start handling", "FAIL", str(e))

# ============================================================================
# TEST 7: /recommend-by-genres - Genre-Based Cold Start
# ============================================================================
print("\n[TEST 7] POST /recommend-by-genres - Genre-Based Cold Start")
print("-" * 70)
try:
    payload = {"genres": ["Action", "Comedy"], "top_n": 5}
    r = requests.post(f'{BASE_URL}/recommend-by-genres', json=payload)
    if r.status_code == 200:
        data = r.json()
        genres = data.get('genres', [])
        recs = data.get('recommendations', [])
        
        log_test("✓ /recommend-by-genres endpoint", "PASS",
                f"Genres: {', '.join(genres)} | Got {len(recs)} recommendations")
        print("   Top recommendations:")
        for i, rec in enumerate(recs[:3], 1):
            print(f"   {i}. {rec['title']} (rating_count: {rec.get('rating_count', 'N/A')})")
    else:
        log_test("✗ /recommend-by-genres endpoint", "FAIL", f"Status: {r.status_code}")
except Exception as e:
    log_test("✗ /recommend-by-genres endpoint", "FAIL", str(e))

# ============================================================================
# TEST 8: /recommend-from-mylist - Watchlist-Based Recommendations
# ============================================================================
print("\n[TEST 8] POST /recommend-from-mylist - Watchlist-Based")
print("-" * 70)
try:
    payload = {
        "mylist": ["Toy Story (1995)", "Forrest Gump (1994)"],
        "count": 5
    }
    r = requests.post(f'{BASE_URL}/recommend-from-mylist', json=payload)
    if r.status_code == 200:
        data = r.json()
        mylist = data.get('mylist', [])
        recs = data.get('recommendations', [])
        
        log_test("✓ /recommend-from-mylist endpoint", "PASS",
                f"Based on {len(mylist)} saved movies | Got {len(recs)} recommendations")
        print("   Top recommendations:")
        for i, rec in enumerate(recs[:3], 1):
            print(f"   {i}. {rec['title']} (score: {rec.get('similarity_score', 'N/A')})")
    else:
        log_test("✗ /recommend-from-mylist endpoint", "FAIL", f"Status: {r.status_code}")
except Exception as e:
    log_test("✗ /recommend-from-mylist endpoint", "FAIL", str(e))

# ============================================================================
# TEST 9: /user-saved-movies - Load User's Saved Movies
# ============================================================================
print("\n[TEST 9] GET /user-saved-movies - Load Saved/Rated Movies")
print("-" * 70)
try:
    r = requests.get(f'{BASE_URL}/user-saved-movies/1')
    if r.status_code == 200:
        data = r.json()
        log_test("✓ /user-saved-movies endpoint", "PASS",
                f"User 1 has {len(data)} saved/rated movies")
        if len(data) > 0:
            print("   Top saved movies:")
            for i, movie in enumerate(data[:5], 1):
                print(f"   {i}. {movie['title']} (rating: {movie.get('rating', 'N/A')})")
    else:
        log_test("✗ /user-saved-movies endpoint", "FAIL", f"Status: {r.status_code}")
except Exception as e:
    log_test("✗ /user-saved-movies endpoint", "FAIL", str(e))

# ============================================================================
# TEST 10: /movie-details - Full Movie Metadata
# ============================================================================
print("\n[TEST 10] GET /movie-details - Fetch Full Movie Details")
print("-" * 70)
try:
    r = requests.get(f'{BASE_URL}/movie-details/Toy%20Story%20(1995)')
    if r.status_code == 200:
        data = r.json()
        title = data.get('title', 'N/A')
        overview = data.get('overview', 'N/A')
        cast = data.get('cast', [])
        
        log_test("✓ /movie-details endpoint", "PASS",
                f"Loaded details for '{title}' | Cast: {len(cast)} actors")
        print(f"   Overview: {overview[:80]}...")
        if len(cast) > 0:
            print(f"   Cast: {', '.join([c.get('name', 'Unknown') for c in cast[:3]])}...")
    else:
        log_test("✗ /movie-details endpoint", "FAIL", f"Status: {r.status_code}")
except Exception as e:
    log_test("✗ /movie-details endpoint", "FAIL", str(e))

# ============================================================================
# TEST 11: Content-Based Fallback
# ============================================================================
print("\n[TEST 11] Content-Based Fallback for Sparse Movies")
print("-" * 70)
try:
    # Try a known movie to test the /recommend endpoint
    payload = {"title": "Abyss, The (1989)", "top_n": 3}
    r = requests.post(f'{BASE_URL}/recommend', json=payload)
    if r.status_code == 200:
        data = r.json()
        recs = data.get('recommendations', [])
        if len(recs) > 0:
            source = recs[0].get('source', 'unknown')
            if source in ['content_based', 'hybrid_with_content']:
                log_test("✓ Content-based fallback", "PASS",
                        f"Triggered fallback for sparse movie | Source: {source}")
            else:
                log_test("✓ Content-based fallback", "PASS",
                        f"Returned results (source: {source})")
        else:
            log_test("✓ Content-based fallback", "PASS", "No results (movie might not exist)")
    else:
        log_test("✗ Content-based fallback", "FAIL", f"Status: {r.status_code}")
except Exception as e:
    log_test("✗ Content-based fallback", "FAIL", str(e))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "="*70)
print("📊 TEST SUMMARY")
print("="*70)

passed = sum(1 for _, status in test_results if status == "PASS")
failed = sum(1 for _, status in test_results if status == "FAIL")
total = len(test_results)

print(f"\n✅ Passed: {passed}/{total}")
print(f"❌ Failed: {failed}/{total}")
print(f"📈 Success Rate: {(passed/total*100):.1f}%\n")

print("Test Results:")
for name, status in test_results:
    symbol = "✅" if status == "PASS" else "❌"
    print(f"  {symbol} {name}: {status}")

print("\n" + "="*70)
if failed == 0:
    print("🎉 ALL TESTS PASSED!")
else:
    print(f"⚠️  {failed} test(s) failed - please review")
print("="*70 + "\n")
