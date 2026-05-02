import { useState, useEffect, useRef } from 'react'
import './App.css'

const EXAMPLE_MOVIES = [
  'Toy Story (1995)',
  'Star Wars (1977)',
  'Fargo (1996)',
  'Pulp Fiction (1994)',
  'Silence of the Lambs, The (1991)',
  'Godfather, The (1972)',
]

function ScoreBar({ score }) {
  const pct = (score * 100).toFixed(1)
  const color =
    score >= 0.75 ? '#f5c518' : score >= 0.5 ? '#e8a030' : '#c0855a'
  return (
    <div className="score-row">
      <div className="score-bar-bg">
        <div
          className="score-bar-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="score-pct">{pct}%</span>
    </div>
  )
}

function MovieCard({ rec }) {
  return (
    <div className="movie-card" style={{ '--delay': `${rec.rank * 60}ms` }}>
      <span className="rank-badge">#{rec.rank}</span>
      <div className="card-body">
        <p className="card-title">{rec.title}</p>
        <div className="score-label">Match strength</div>
        <ScoreBar score={rec.similarity_score} />
      </div>
    </div>
  )
}

function HowItWorks() {
  const steps = [
    { icon: '🎬', label: 'Pick a movie', desc: 'Find any title in our curated list' },
    { icon: '⚙️', label: 'AI Magic', desc: 'We scan 100k ratings for patterns' },
    { icon: '🍿', label: 'Get Picks', desc: 'Top 10 films ranked by overlap' },
  ]
  return (
    <div className="how-it-works">
      {steps.map((s, i) => (
        <div key={i} className="step">
          <div className="step-icon">{s.icon}</div>
          <div className="step-label">{s.label}</div>
          <div className="step-desc">{s.desc}</div>
        </div>
      ))}
    </div>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [results, setResults] = useState(null)
  const [mode, setMode] = useState('item') // 'item' | 'hybrid'
  const [userId, setUserId] = useState('')
  const [userIdError, setUserIdError] = useState(null)

  // Autocomplete states
  const [allMovies, setAllMovies] = useState([])
  const [suggestions, setSuggestions] = useState([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState(-1)
  const [isFetchingMovies, setIsFetchingMovies] = useState(true)

  const searchContainerRef = useRef(null)

  // Fetch all available movies on load
  useEffect(() => {
    const fetchMovies = async () => {
      try {
        const res = await fetch('http://localhost:5000/movies')
        const data = await res.json()
        if (Array.isArray(data)) setAllMovies(data)
      } catch (err) {
        console.error('Failed to fetch movies:', err)
      } finally {
        setIsFetchingMovies(false)
      }
    }
    fetchMovies()
  }, [])

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchContainerRef.current && !searchContainerRef.current.contains(event.target)) {
        setShowSuggestions(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const search = async (title) => {
    const q = title ?? query
    if (!q.trim()) return

    // Validate user ID when in hybrid mode
    if (mode === 'hybrid') {
      const uid = parseInt(userId, 10)
      if (!userId.trim() || isNaN(uid) || uid < 1) {
        setUserIdError('Please enter a valid User ID (1–943)')
        return
      }
      setUserIdError(null)
    }

    setQuery(q)
    setLoading(true)
    setError(null)
    setResults(null)
    setShowSuggestions(false)

    try {
      if (mode === 'item') {
        const res = await fetch('http://localhost:5000/recommend', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: q, top_n: 10 }),
        })
        const data = await res.json()
        if (!res.ok) setError(data.error || 'Something went wrong')
        else setResults(data)
      } else {
        const res = await fetch('http://localhost:5000/recommend-hybrid', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            title: q,
            user_id: parseInt(userId, 10),
            top_n: 10,
            item_weight: 0.5,
            user_weight: 0.5,
          }),
        })
        const data = await res.json()
        if (!res.ok) setError(data.error || 'Something went wrong')
        else setResults(data)
      }
    } catch {
      setError('Could not reach the server. Make sure app.py is running on port 5000.')
    } finally {
      setLoading(false)
    }
  }

  const handleInputChange = (e) => {
    const val = e.target.value
    setQuery(val)

    if (val.trim()) {
      const filtered = allMovies
        .filter(m => m.toLowerCase().includes(val.toLowerCase()))
        .slice(0, 10)
      setSuggestions(filtered)
      setShowSuggestions(true)
      setSelectedIndex(-1)
    } else {
      setSuggestions([])
      setShowSuggestions(false)
    }
  }

  const handleKeyDown = (e) => {
    if (!showSuggestions) return

    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelectedIndex(prev => (prev < suggestions.length - 1 ? prev + 1 : prev))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelectedIndex(prev => (prev > 0 ? prev - 1 : prev))
    } else if (e.key === 'Enter') {
      if (selectedIndex >= 0) {
        e.preventDefault()
        handleSelect(suggestions[selectedIndex])
      }
    } else if (e.key === 'Escape') {
      setShowSuggestions(false)
    }
  }

  const handleSelect = (title) => {
    setQuery(title)
    setShowSuggestions(false)
    search(title)
  }

  const randomizeUserId = () => {
    const randomId = Math.floor(Math.random() * 943) + 1
    setUserId(randomId.toString())
    setUserIdError(null)
  }

  const switchMode = (newMode) => {
    setMode(newMode)
    setResults(null)
    setError(null)
    setUserIdError(null)
  }

  return (
    <div className="app">
      {/* ── Hero ── */}
      <header className="hero">
        <div className="hero-eyebrow">Hybrid Movie Recommender</div>
        <h1 className="hero-title">
          Find your next<br />
          <span className="hero-accent">favourite film.</span>
        </h1>
        <p className="hero-sub">
          Using Item-Item and User-Based Collaborative Filtering to find movies
          you'll love based on shared audience taste.
        </p>
      </header>

      {/* ── How it works ── */}
      {!results && !loading && <HowItWorks />}

      {/* ── Mode Selection ── */}
      <div className="mode-section">
        <div className="mode-toggle">
          <button
            className={`mode-btn ${mode === 'item' ? 'active' : ''}`}
            onClick={() => switchMode('item')}
          >
            🎬 Movie Match
          </button>
          <button
            className={`mode-btn ${mode === 'hybrid' ? 'active' : ''}`}
            onClick={() => switchMode('hybrid')}
          >
            ✨ For You
          </button>
        </div>
        
        <div className="mode-description">
          {mode === 'item' ? (
            <>
              <span className="mode-headline">Movie Match Mode:</span>
              <span className="mode-desc"> Finds movies that are statistically similar to the one you type. No login needed.</span>
            </>
          ) : (
            <>
              <span className="mode-headline">For You Mode:</span>
              <span className="mode-desc"> Blends the movie's similarity with your personal rating history to find a unique match.</span>
            </>
          )}
        </div>
      </div>

      {/* ── Search Section ── */}
      <div className="search-section" ref={searchContainerRef}>
        <form className="search-form" onSubmit={(e) => { e.preventDefault(); search() }}>
          
          {/* User ID input — only visible in hybrid mode */}
          {mode === 'hybrid' && (
            <div className="uid-group">
              <label className="input-label">
                Viewer ID <span className="label-note">(Range: 1–943)</span>
              </label>
              <div className="uid-row">
                <input
                  type="number"
                  className={`search-input uid-input ${userIdError ? 'input-error' : ''}`}
                  placeholder="ID..."
                  value={userId}
                  onChange={(e) => { setUserId(e.target.value); setUserIdError(null) }}
                />
                <button type="button" className="random-btn" onClick={randomizeUserId}>
                  🎲 Random ID
                </button>
              </div>
              {userIdError && <span className="field-error">{userIdError}</span>}
              <p className="field-hint">
                In this demo, we use historical IDs from the MovieLens dataset.
              </p>
            </div>
          )}

          <div className="input-group">
            <label className="input-label">Search Movie</label>
            <div className="search-row">
              <div className="input-wrapper">
                <input
                  type="text"
                  className={`search-input ${isFetchingMovies ? 'input-loading' : ''}`}
                  placeholder={isFetchingMovies ? 'Loading database...' : 'e.g. Pulp Fiction (1994)'}
                  value={query}
                  onChange={handleInputChange}
                  onKeyDown={handleKeyDown}
                  onFocus={() =>
                    query.trim() &&
                    setSuggestions(allMovies.filter(m => m.toLowerCase().includes(query.toLowerCase())).slice(0, 10)) &&
                    setShowSuggestions(true)
                  }
                  disabled={isFetchingMovies}
                />

                {showSuggestions && (
                  <div className="suggestions-dropdown">
                    {suggestions.length > 0 ? (
                      suggestions.map((s, i) => (
                        <div
                          key={s}
                          className={`suggestion-item ${i === selectedIndex ? 'active' : ''}`}
                          onClick={() => handleSelect(s)}
                        >
                          {s}
                        </div>
                      ))
                    ) : (
                      <div className="suggestion-empty">No results found.</div>
                    )}
                  </div>
                )}
              </div>

              <button className="search-btn" type="submit" disabled={loading || isFetchingMovies}>
                {loading ? <span className="spinner" /> : 'Find Similar'}
              </button>
            </div>
          </div>
        </form>

        {!results && (
          <div className="examples">
            <span className="examples-label">Quick search:</span>
            {EXAMPLE_MOVIES.map((m) => (
              <button
                key={m}
                className="chip"
                onClick={() => handleSelect(m)}
                disabled={loading || isFetchingMovies}
              >
                {m}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* ── Status Messages ── */}
      {error && (
        <div className="error-box">
          <span className="error-icon">⚠</span>
          <span>{error}</span>
        </div>
      )}

      {loading && (
        <div className="cards-grid">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="movie-card skeleton" />
          ))}
        </div>
      )}

      {/* ── Results ── */}
      {results && (
        <div className="results">
          <div className="results-header">
            <div>
              <h2 className="results-title">
                {results.mode === 'hybrid'
                  ? <>For User <em>#{results.user_id}</em></>
                  : <>Matches for <em>"{results.movie}"</em></>
                }
              </h2>
              {results.mode === 'hybrid' && (
                <p className="results-sub">Blended with your similarity to "{results.movie}"</p>
              )}
            </div>
            <button className="reset-btn" onClick={() => { setResults(null); setQuery('') }}>
              ← Reset search
            </button>
          </div>

          <div className="cards-grid">
            {results.recommendations.map((rec) => (
              <MovieCard key={rec.rank} rec={rec} />
            ))}
          </div>
          
          <div className="info-box">
            <span className="info-icon">ℹ</span>
            <div>
              <strong>Pro Tip:</strong> We filter for movies with at least 50 ratings to ensure high-quality recommendations.
            </div>
          </div>
        </div>
      )}

      <footer className="footer">
        <p>Movie Recommendation System &mdash; Syed Asjad Ali Zaidi · Affan Jan · Muhammad Saad</p>
        <p>Dataset: <a href="https://grouplens.org/datasets/movielens/100k/" target="_blank" rel="noreferrer">MovieLens 100K</a></p>
      </footer>
    </div>
  )
}