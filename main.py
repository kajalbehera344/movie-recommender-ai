import asyncio
import datetime
import json
import os
import pickle
import re
import time
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


# =========================
# ENV
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    # Don't crash import-time in production if you prefer; but for you better fail early:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")

# Chatbot (optional): Groq's OpenAI-compatible API. Without a key, /chat returns 503.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY and GROQ_API_KEY.startswith("your_"):
    GROQ_API_KEY = None
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Tried in order. Each model has its own free-tier limit (8K tokens/min), so falling back
# when one is rate limited or retired keeps the chatbot working. Override with GROQ_MODEL.
GROQ_MODELS = [
    m for m in [os.getenv("GROQ_MODEL"), "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"] if m
]


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PICKLE GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")
# IMDb's public ratings dataset (datasets.imdbws.com), filtered to titles with >= 25 votes
IMDB_RATINGS_PATH = os.path.join(BASE_DIR, "imdb_ratings.tsv.gz")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None

# One shared client so TMDB calls reuse connections instead of a new TLS handshake each time
http_client: Optional[httpx.AsyncClient] = None

# Small in-memory cache of TMDB responses: (path, params) -> (fetched_at, json)
TMDB_CACHE_TTL = 600
TMDB_CACHE_MAX = 2000
_tmdb_cache: Dict[Tuple, Tuple[float, Dict[str, Any]]] = {}
# Caps parallel TMDB requests (a details page with IMDb ratings fans out to ~40 calls)
_tmdb_slots = asyncio.Semaphore(20)

# IMDb ratings as sorted numeric arrays for fast, low-memory lookup (tt0111161 -> 111161)
imdb_ids: Optional[np.ndarray] = None
imdb_scores: Optional[np.ndarray] = None
imdb_votes: Optional[np.ndarray] = None


# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None
    overview: Optional[str] = None
    imdb_id: Optional[str] = None
    imdb_rating: Optional[float] = None
    imdb_votes: Optional[int] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []
    runtime: Optional[int] = None
    vote_average: Optional[float] = None
    imdb_id: Optional[str] = None
    imdb_rating: Optional[float] = None
    imdb_votes: Optional[int] = None


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe TMDB GET:
    - Network errors -> 502
    - TMDB API errors -> 502 with detail
    - Successful responses cached for TMDB_CACHE_TTL seconds
    """
    cache_key = (path, tuple(sorted(params.items())))
    hit = _tmdb_cache.get(cache_key)
    if hit and time.monotonic() - hit[0] < TMDB_CACHE_TTL:
        return hit[1]

    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with _tmdb_slots:
            r = await http_client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"TMDB error {r.status_code}: {r.text}"
        )

    data = r.json()
    if len(_tmdb_cache) >= TMDB_CACHE_MAX:
        _tmdb_cache.clear()
    _tmdb_cache[cache_key] = (time.monotonic(), data)
    return data


# =========================
# IMDb RATINGS
# =========================
def load_imdb_ratings() -> None:
    """Load the bundled IMDb ratings file; if it's missing, ratings are just left empty."""
    global imdb_ids, imdb_scores, imdb_votes
    if not os.path.exists(IMDB_RATINGS_PATH):
        print(f"[warn] {IMDB_RATINGS_PATH} not found - IMDb ratings disabled")
        return
    r = pd.read_csv(IMDB_RATINGS_PATH, sep="\t", dtype={"tconst": str})
    ids = r["tconst"].str[2:].astype(np.int64).to_numpy()
    order = np.argsort(ids)
    imdb_ids = ids[order]
    imdb_scores = r["averageRating"].to_numpy(dtype=np.float32)[order]
    imdb_votes = r["numVotes"].to_numpy(dtype=np.int64)[order]


def imdb_rating(imdb_id: Optional[str]) -> Tuple[Optional[float], Optional[int]]:
    """(rating, votes) for an IMDb id like 'tt0111161', or (None, None) if unrated."""
    if imdb_ids is None or not imdb_id or not imdb_id.startswith("tt"):
        return None, None
    try:
        n = int(imdb_id[2:])
    except ValueError:
        return None, None
    i = int(np.searchsorted(imdb_ids, n))
    if i < len(imdb_ids) and imdb_ids[i] == n:
        return round(float(imdb_scores[i]), 1), int(imdb_votes[i])
    return None, None


async def add_imdb_to_card(card: TMDBMovieCard) -> TMDBMovieCard:
    """TMDB list results don't include the IMDb id, so look it up from the (cached) details."""
    try:
        data = await tmdb_get(f"/movie/{card.tmdb_id}", {"language": "en-US"})
        card.imdb_id = data.get("imdb_id") or None
        card.imdb_rating, card.imdb_votes = imdb_rating(card.imdb_id)
    except Exception:
        pass
    return card


async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
                overview=m.get("overview"),
            )
        )
    await asyncio.gather(*(add_imdb_to_card(c) for c in out))
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    imdb_id = data.get("imdb_id") or None
    rating, votes = imdb_rating(imdb_id)
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
        runtime=data.get("runtime"),
        vote_average=data.get("vote_average"),
        imdb_id=imdb_id,
        imdb_rating=rating,
        imdb_votes=votes,
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """
    Raw TMDB response for keyword search (MULTIPLE results).
    Streamlit will use this for suggestions and grid.
    """
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


async def tmdb_discover_by_first_genre(
    details: TMDBMovieDetails, limit: int
) -> List[TMDBMovieCard]:
    """Popular movies in the movie's first genre, excluding the movie itself."""
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != details.tmdb_id]


# =========================
# TF-IDF Helpers
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices.pkl can be:
    - dict(title -> index)
    - pandas Series (index=title, value=index)
    We normalize into TITLE_TO_IDX.
    """
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    # pandas Series or similar mapping
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        # last resort: if it's a list-like etc.
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        )


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Returns list of (title, score) from local df using cosine similarity on TF-IDF matrix.
    Safe against missing columns/rows.
    """
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)

    # query vector
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    # sort descending
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """
    Uses TMDB search by title to fetch poster for a local title.
    If not found, returns None (never crashes the endpoint).
    """
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        card = TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
            overview=m.get("overview"),
        )
        return await add_imdb_to_card(card)
    except Exception:
        return None


# =========================
# STARTUP: LOAD PICKLES
# =========================
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    # Load df
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    # Load indices
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    # Load TF-IDF matrix (usually scipy sparse)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Load tfidf vectorizer (optional, not used directly here)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    # Build normalized map
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    # sanity
    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")

    load_imdb_ratings()


@app.on_event("startup")
async def open_http_client():
    global http_client
    http_client = httpx.AsyncClient(timeout=20)


@app.on_event("shutdown")
async def close_http_client():
    if http_client is not None:
        await http_client.aclose()


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- HOME FEED (TMDB) ----------
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """
    Home feed for Streamlit (posters).
    category:
      - trending (trending/movie/day)
      - popular, top_rated, upcoming, now_playing  (movie/{category})
    """
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


# ---------- TMDB KEYWORD SEARCH (MULTIPLE RESULTS) ----------
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """
    Returns RAW TMDB shape with 'results' list.
    Streamlit will use it for:
      - dropdown suggestions
      - grid results
    """
    return await tmdb_search_movies(query=query, page=page)


# ---------- MOVIE DETAILS (SAFE ROUTE) ----------
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


# ---------- GENRE RECOMMENDATIONS ----------
@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    """
    Given a TMDB movie ID:
    - fetch details
    - pick first genre
    - discover movies in that genre (popular)
    """
    details = await tmdb_movie_details(tmdb_id)
    return await tmdb_discover_by_first_genre(details, limit)


# ---------- TF-IDF ONLY (debug/useful) ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


# ---------- BUNDLE: Details + TF-IDF recs + Genre recs ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    This endpoint is for when you have a selected movie and want:
      - movie details
      - TF-IDF recommendations (local) + posters
      - Genre recommendations (TMDB) + posters

    NOTE:
    - It selects the BEST match from TMDB for the given query.
    - If you want MULTIPLE matches, use /tmdb/search
    """
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No TMDB movie found for query: {query}"
        )

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # 1) TF-IDF recommendations (never crash endpoint)
    recs: List[Tuple[str, float]] = []
    try:
        # try local dataset by TMDB title
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        # fallback to user query
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    # 2) Poster lookups for TF-IDF recs + genre recs (TMDB discover), all in parallel
    tfidf_cards, genre_recs = await asyncio.gather(
        asyncio.gather(*(attach_tmdb_card_by_title(title) for title, _ in recs)),
        tmdb_discover_by_first_genre(details, genre_limit),
    )
    tfidf_items = [
        TFIDFRecItem(title=title, score=score, tmdb=card)
        for (title, score), card in zip(recs, tfidf_cards)
    ]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )


# =========================
# CHATBOT (Groq LLM + tools over TMDB / IMDb / TF-IDF)
# =========================
# The LLM only decides which tool to call and writes the answer; every movie, year and
# rating comes from the tools, so it can't invent titles or ratings.
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    reply: str
    movies: List[TMDBMovieCard] = []


CHAT_SYSTEM_PROMPT = """You are CineBot, the movie assistant inside a Movie Recommender app. Today is {today}.
- Use the tools for every list, rating or fact about movies. Never invent titles, years or ratings; only use what the tools return.
- If the user doesn't say how many movies, use 5 (maximum 10).
- "top"/"best"/"highest rated" -> sort_by top_rated; "popular" -> popular; "new"/"latest"/"recent" -> newest; "trending"/"this week" -> trending_movies.
- Reply with a short numbered list: **Title** (Year) - IMDb 8.1/10 - one short line on why it's worth watching. Write "IMDb: not rated" when the rating is missing.
- Posters are shown automatically under your reply, so don't add links, image URLs or tables.
- If a tool returns no movies, say so and suggest a broader search.
- For questions that aren't about movies, politely say you can only help with movies."""

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_movies",
            "description": (
                "Find movies filtered by genre, release years, original language and/or a person "
                "(actor or director), sorted by rating, popularity or release date. Use for questions "
                "like 'top 5 horror movies', 'best Hindi comedies', 'Tom Hanks movies', 'new sci-fi films'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "genres": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Genre names, all must match: Action, Adventure, Animation, Comedy, Crime, "
                            "Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, "
                            "Science Fiction, Thriller, War, Western"
                        ),
                    },
                    "year_from": {"type": "integer", "description": "Earliest release year, e.g. 2010"},
                    "year_to": {"type": "integer", "description": "Latest release year, e.g. 2019"},
                    "language": {
                        "type": "string",
                        "description": (
                            "ISO 639-1 original language: hi = Hindi/Bollywood, ta = Tamil, te = Telugu, "
                            "ml = Malayalam, ko = Korean, ja = Japanese, fr = French, es = Spanish, en = English"
                        ),
                    },
                    "person": {"type": "string", "description": "Actor or director name"},
                    "sort_by": {"type": "string", "enum": ["top_rated", "popular", "newest"]},
                    "limit": {"type": "integer", "description": "Number of movies, 1-10 (default 5)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_movie_details",
            "description": (
                "Facts about one movie: year, genres, runtime, IMDb rating, director, main cast and plot. "
                "Use for 'tell me about X', 'who directed X', 'what is the IMDb rating of X'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "year": {"type": "integer", "description": "Release year, if known, to pick the right movie"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "similar_movies",
            "description": "Movies similar to a given movie (uses the app's TF-IDF recommendation model). Use for 'movies like X'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "limit": {"type": "integer", "description": "Number of movies, 1-10 (default 5)"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trending_movies",
            "description": "Movies trending right now.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {"type": "string", "enum": ["day", "week"]},
                    "limit": {"type": "integer", "description": "Number of movies, 1-10 (default 5)"},
                },
            },
        },
    },
]

GENRE_ALIASES = {
    "sci-fi": "science fiction", "scifi": "science fiction", "sf": "science fiction",
    "rom-com": "romance", "romcom": "romance", "romantic": "romance",
    "kids": "family", "cartoon": "animation", "anime": "animation", "animated": "animation",
    "scary": "horror", "funny": "comedy", "suspense": "thriller", "historical": "history",
    "musical": "music", "detective": "mystery",
}


def _clamp_limit(limit: Any, default: int = 5) -> int:
    try:
        return max(1, min(int(limit), 10))
    except (TypeError, ValueError):
        return default


def _card_summary(c: TMDBMovieCard) -> Dict[str, Any]:
    """Compact movie info passed back to the LLM."""
    return {
        "title": c.title,
        "year": (c.release_date or "")[:4] or None,
        "imdb_rating": c.imdb_rating,
        "imdb_votes": c.imdb_votes,
        "tmdb_rating": c.vote_average,
        "overview": (c.overview or "")[:120],
    }


def _norm_text(s: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() + " "


def _movies_in_reply(cards: List[TMDBMovieCard], reply: str) -> List[TMDBMovieCard]:
    """Posters for the movies the reply actually names, in the order it names them."""
    text = _norm_text(reply)
    hits = [(text.find(_norm_text(c.title)), c) for c in cards]
    hits = [(pos, c) for pos, c in hits if pos >= 0]
    return [c for _, c in sorted(hits, key=lambda h: h[0])] if hits else cards


async def _genre_ids(names: List[str]) -> Tuple[List[int], List[str]]:
    data = await tmdb_get("/genre/movie/list", {"language": "en-US"})
    lookup = {g["name"].lower(): g["id"] for g in data.get("genres", [])}
    ids, unknown = [], []
    for n in names:
        key = GENRE_ALIASES.get(n.strip().lower(), n.strip().lower())
        if key in lookup:
            ids.append(lookup[key])
        else:
            unknown.append(n)
    return ids, unknown


async def tool_find_movies(
    genres: Any = None,
    year_from: Any = None,
    year_to: Any = None,
    language: Optional[str] = None,
    person: Optional[str] = None,
    sort_by: str = "top_rated",
    limit: Any = 5,
) -> Tuple[Dict[str, Any], List[TMDBMovieCard]]:
    limit = _clamp_limit(limit)
    today = datetime.date.today().isoformat()
    params: Dict[str, Any] = {"language": "en-US", "include_adult": "false", "page": 1}
    notes: List[str] = []

    if isinstance(genres, str):
        genres = [genres]
    if genres:
        ids, unknown = await _genre_ids(genres)
        if ids:
            params["with_genres"] = ",".join(str(i) for i in ids)
        if unknown:
            notes.append(f"unknown genres ignored: {', '.join(unknown)}")
    if year_from:
        params["primary_release_date.gte"] = f"{int(year_from)}-01-01"
    if year_to:
        params["primary_release_date.lte"] = f"{int(year_to)}-12-31"
    if language:
        params["with_original_language"] = language.strip().lower()[:2]
    if person:
        found = await tmdb_get("/search/person", {"query": person, "include_adult": "false"})
        people = found.get("results") or []
        if not people:
            return {"movies": [], "note": f"No person found named '{person}'"}, []
        params["with_people"] = people[0]["id"]
        notes.append(f"person matched: {people[0].get('name')}")

    if sort_by == "popular":
        params["sort_by"] = "popularity.desc"
    elif sort_by == "newest":
        params["sort_by"] = "primary_release_date.desc"
        params["primary_release_date.lte"] = min(params.get("primary_release_date.lte", today), today)
        params["vote_count.gte"] = 20
    else:
        # Rating sort needs a vote floor, or obscure films with three 10/10 votes come first
        narrow = bool(person or year_from or year_to or (language and language[:2] != "en"))
        params["sort_by"] = "vote_average.desc"
        params["vote_count.gte"] = 200 if narrow else 1000

    data = await tmdb_get("/discover/movie", params)
    results = data.get("results") or []
    if not results and "vote_count.gte" in params:
        params["vote_count.gte"] = 10
        results = (await tmdb_get("/discover/movie", params)).get("results") or []

    cards = await tmdb_cards_from_results(results, limit=20)
    if sort_by not in ("popular", "newest"):
        # "Top" = best IMDb rating among TMDB's top-rated candidates
        cards.sort(key=lambda c: (c.imdb_rating is not None, c.imdb_rating or 0.0), reverse=True)
    cards = cards[:limit]

    payload: Dict[str, Any] = {"movies": [_card_summary(c) for c in cards]}
    if notes:
        payload["note"] = "; ".join(notes)
    return payload, cards


async def tool_get_movie_details(
    title: str, year: Any = None
) -> Tuple[Dict[str, Any], List[TMDBMovieCard]]:
    params: Dict[str, Any] = {"query": title, "include_adult": "false", "language": "en-US", "page": 1}
    if year:
        params["primary_release_year"] = int(year)
    results = (await tmdb_get("/search/movie", params)).get("results") or []
    if not results:
        return {"error": f"No movie found for '{title}'"}, []

    tmdb_id = int(results[0]["id"])
    details, credits = await asyncio.gather(
        tmdb_movie_details(tmdb_id),
        tmdb_get(f"/movie/{tmdb_id}/credits", {"language": "en-US"}),
    )
    directors = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"]
    cast = [c["name"] for c in credits.get("cast", [])[:5]]

    card = TMDBMovieCard(
        tmdb_id=details.tmdb_id,
        title=details.title,
        poster_url=details.poster_url,
        release_date=details.release_date,
        vote_average=details.vote_average,
        overview=details.overview,
        imdb_id=details.imdb_id,
        imdb_rating=details.imdb_rating,
        imdb_votes=details.imdb_votes,
    )
    payload = {
        "title": details.title,
        "release_date": details.release_date,
        "genres": [g.get("name") for g in details.genres],
        "runtime_minutes": details.runtime,
        "imdb_rating": details.imdb_rating,
        "imdb_votes": details.imdb_votes,
        "tmdb_rating": details.vote_average,
        "directors": directors,
        "cast": cast,
        "overview": details.overview,
    }
    return payload, [card]


async def tool_similar_movies(
    title: str, limit: Any = 5
) -> Tuple[Dict[str, Any], List[TMDBMovieCard]]:
    limit = _clamp_limit(limit)
    best = await tmdb_search_first(title)

    # Prefer the app's own TF-IDF model; try the user's wording, then TMDB's canonical title
    recs: List[Tuple[str, float]] = []
    for candidate in [title] + ([best["title"]] if best and best.get("title") else []):
        try:
            recs = tfidf_recommend_titles(candidate, top_n=limit)
            break
        except HTTPException:
            continue

    if recs:
        found = await asyncio.gather(*(attach_tmdb_card_by_title(t) for t, _ in recs))
        cards = [c for c in found if c is not None]
        source = "TF-IDF content-based model"
    elif best:
        data = await tmdb_get(f"/movie/{int(best['id'])}/recommendations", {"language": "en-US", "page": 1})
        cards = await tmdb_cards_from_results(data.get("results", []), limit=limit)
        source = "TMDB recommendations (movie not in the local dataset)"
    else:
        return {"error": f"No movie found for '{title}'"}, []

    return {"source": source, "movies": [_card_summary(c) for c in cards]}, cards


async def tool_trending_movies(
    period: str = "week", limit: Any = 5
) -> Tuple[Dict[str, Any], List[TMDBMovieCard]]:
    period = period if period in ("day", "week") else "week"
    data = await tmdb_get(f"/trending/movie/{period}", {"language": "en-US"})
    cards = await tmdb_cards_from_results(data.get("results", []), limit=_clamp_limit(limit))
    return {"movies": [_card_summary(c) for c in cards]}, cards


CHAT_TOOL_IMPLS = {
    "find_movies": tool_find_movies,
    "get_movie_details": tool_get_movie_details,
    "similar_movies": tool_similar_movies,
    "trending_movies": tool_trending_movies,
}


async def run_chat_tool(call: Dict[str, Any]) -> Tuple[Dict[str, Any], List[TMDBMovieCard]]:
    """Run one tool call from the LLM; errors go back to the LLM instead of failing the request."""
    fn = call.get("function") or {}
    impl = CHAT_TOOL_IMPLS.get(fn.get("name"))
    if impl is None:
        return {"error": f"unknown tool {fn.get('name')}"}, []
    try:
        args = json.loads(fn.get("arguments") or "{}") or {}
    except json.JSONDecodeError:
        return {"error": "tool arguments were not valid JSON"}, []

    allowed = impl.__code__.co_varnames[: impl.__code__.co_argcount]
    kwargs = {k: v for k, v in args.items() if k in allowed and v not in (None, "")}
    try:
        return await impl(**kwargs)
    except HTTPException as e:
        return {"error": str(e.detail)[:300]}, []
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:300]}, []


def _groq_model_options(model: str) -> Dict[str, Any]:
    # Reasoning models: keep hidden "thinking" short, since it counts against the token limit
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    if model.startswith("qwen/"):
        return {"reasoning_format": "hidden"}
    return {}


# Models that hit their rate limit are skipped until this time (monotonic seconds)
_groq_cooldown: Dict[str, float] = {}


async def groq_complete(messages: List[Dict[str, Any]], allow_tools: bool = True) -> Dict[str, Any]:
    """One Groq chat completion. Rate-limited, retired or failing models fall back to the next one."""
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Chatbot is not set up: add GROQ_API_KEY=... to .env "
            "(free key at https://console.groq.com/keys) and restart the backend.",
        )

    now = time.monotonic()
    models = [m for m in GROQ_MODELS if _groq_cooldown.get(m, 0) <= now] or GROQ_MODELS
    errors: List[str] = []
    rate_limited = False
    for model in models:
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_completion_tokens": 1024,
            "tools": CHAT_TOOLS,
            "tool_choice": "auto" if allow_tools else "none",
            **_groq_model_options(model),
        }
        for attempt in range(2):
            try:
                r = await http_client.post(
                    GROQ_URL,
                    json=body,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    timeout=60,
                )
            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail=f"Could not reach Groq: {type(e).__name__}")

            if r.status_code == 200:
                data = r.json()
                print(f"[chat] {model} used {data.get('usage', {}).get('total_tokens')} tokens")
                return data["choices"][0]["message"]
            if r.status_code == 401:
                raise HTTPException(status_code=502, detail="Groq rejected the API key (401). Check GROQ_API_KEY in .env.")
            if r.status_code == 429:
                try:
                    wait = float(r.headers.get("retry-after", "10"))
                except ValueError:
                    wait = 10.0
                if wait <= 3 and attempt == 0:
                    await asyncio.sleep(wait)
                    continue
                _groq_cooldown[model] = time.monotonic() + wait
                rate_limited = True
                break
            # Occasionally a model emits a malformed tool call; one retry usually fixes it
            if r.status_code == 400 and "tool_use_failed" in r.text and attempt == 0:
                continue
            break

        errors.append(f"{model}: HTTP {r.status_code} {r.text[:200]}")

    if rate_limited:
        raise HTTPException(
            status_code=503,
            detail="The chatbot is busy (Groq free-tier limit of 8,000 tokens per minute). "
            "Please wait about a minute and ask again.",
        )
    raise HTTPException(status_code=502, detail="Groq error - " + " | ".join(errors)[:600])


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Last few turns only: enough for follow-ups ("only ones after 2010") without burning tokens
    history = [
        {"role": m.role, "content": m.content[:1500]}
        for m in req.messages[-6:]
        if m.role in ("user", "assistant") and m.content.strip()
    ]
    if not history or history[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="The last message must be from the user")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT.format(today=datetime.date.today().isoformat())}
    ] + history
    shown: List[TMDBMovieCard] = []

    msg: Dict[str, Any] = {}
    for round_no in range(4):
        # After 3 tool rounds, make the model answer with what it has
        msg = await groq_complete(messages, allow_tools=round_no < 3)
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        results = await asyncio.gather(*(run_chat_tool(c) for c in calls))
        for call, (payload, cards) in zip(calls, results):
            shown.extend(cards)
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            })

    seen, movies = set(), []
    for c in shown:
        if c.tmdb_id not in seen:
            seen.add(c.tmdb_id)
            movies.append(c)

    reply = (msg.get("content") or "").strip() or "Sorry, I couldn't come up with an answer. Try rephrasing?"
    return ChatResponse(reply=reply, movies=_movies_in_reply(movies, reply)[:10])