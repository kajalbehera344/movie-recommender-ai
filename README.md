# 🎬 Movie Recommender System

**Content-based movie recommendations powered by NLP, TF-IDF & FastAPI, with a Streamlit front end.**

This project is a full-stack movie recommendation system. A **FastAPI** backend serves content-based recommendations computed offline with **TF-IDF + cosine similarity**, enriches them with live data from **TMDB (The Movie Database)**, and a **Streamlit** app provides a browsable UI for searching movies, viewing details, and exploring similar titles.

🔗 **Live demo:** [[movie-recommender-system]((https://movie-rec-frontend-1wjy.onrender.com/)

---

## ✨ Features

- **🔎 Keyword search with autocomplete** — type a title and get live TMDB suggestions plus a poster grid of matches.
- **🏠 Home feed** — browse Trending, Popular, Top Rated, Now Playing, or Upcoming movies.
- **📄 Movie details page** — poster, backdrop, overview, release date, and genres for any title.
- **🧠 Content-based recommendations (TF-IDF)** — similar movies computed from a locally trained TF-IDF matrix + cosine similarity over movie metadata.
- **🎭 Genre-based recommendations** — a secondary "More Like This" feed pulled from TMDB's discover endpoint using the movie's primary genre.
- **⚡ FastAPI backend** — a documented REST API (`/docs` via Swagger UI) that the frontend (or any client) can consume.

---

## 🧩 Architecture

```
┌──────────────────┐        HTTP        ┌───────────────────┐        HTTP        ┌──────────────┐
│  Streamlit App    │  ───────────────▶ │   FastAPI Backend   │  ───────────────▶ │   TMDB API    │
│    (app.py)        │ ◀─────────────── │      (main.py)       │ ◀─────────────── │ (posters,     │
└──────────────────┘                    └───────────┬─────────┘                    │  metadata)    │
                                                       │                             └──────────────┘
                                                       │ loads at startup
                                                       ▼
                                     df.pkl · indices.pkl · tfidf.pkl · tfidf_matrix.pkl
                                     (precomputed TF-IDF model, built in the notebook)
```

- **`main.py`** — FastAPI service. Loads the precomputed TF-IDF artifacts on startup, proxies/enriches data from TMDB, and exposes recommendation endpoints.
- **`app.py`** — Streamlit UI. Calls the FastAPI backend to render the home feed, search results, and a movie details + recommendations page.
- **`Movie_Recommender_System.ipynb`** — the notebook used to clean the movie dataset, build the TF-IDF vectorizer/matrix, and export the `.pkl` artifacts consumed by `main.py`.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI, Uvicorn |
| Recommendation engine | Scikit-learn (TF-IDF), SciPy (sparse matrix ops), NumPy, Pandas |
| External data | TMDB API (`httpx` async client) |
| Frontend | Streamlit |
| Config | python-dotenv |
| Deployment | Render (`render.yaml`), Vercel |

---

## 📁 Project Structure

```
Movie-Recommender-System/
├── app.py                        # Streamlit frontend
├── main.py                       # FastAPI backend (REST API)
├── Movie_Recommender_System.ipynb # Data prep + TF-IDF model training notebook
├── df.pkl / df.zip                # Cleaned movie metadata DataFrame
├── indices.pkl / indices.zip      # Title → row-index lookup map
├── tfidf.pkl                      # Fitted TF-IDF vectorizer
├── tfidf_matrix.pkl               # TF-IDF feature matrix (movie vectors)
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Python runtime version
├── render.yaml                    # Render deployment config (API + frontend services)
├── .devcontainer/                 # Dev container config
└── .gitignore
```

---

## 📡 API Reference

Base URL (local): `http://127.0.0.1:8000` · Interactive docs at `/docs`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/home?category=popular&limit=24` | Home feed — `trending`, `popular`, `top_rated`, `now_playing`, or `upcoming` |
| `GET` | `/tmdb/search?query=...&page=1` | Raw TMDB keyword search (used for autocomplete + result grid) |
| `GET` | `/movie/id/{tmdb_id}` | Full details for a single movie by TMDB ID |
| `GET` | `/recommend/tfidf?title=...&top_n=10` | TF-IDF–only similar-title recommendations |
| `GET` | `/recommend/genre?tmdb_id=...&limit=18` | Genre-based recommendations via TMDB discover |
| `GET` | `/movie/search?query=...&tfidf_top_n=12&genre_limit=12` | Bundle: movie details + TF-IDF recs + genre recs in one call |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (the project targets `3.10.13`, see `runtime.txt` / `render.yaml`)
- A free [TMDB API key](https://www.themoviedb.org/settings/api)

### 1. Clone the repo

```bash
git clone https://github.com/Aayush-0409/Movie-Recommender-System.git
cd Movie-Recommender-System
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
TMDB_API_KEY=your_tmdb_api_key_here
```

> The backend raises an error at startup if `TMDB_API_KEY` isn't set.

### 4. Unzip the model artifacts (if needed)

The TF-IDF model files are provided as both `.pkl` and `.zip`. If the `.pkl` files aren't present, unzip them first:

```bash
unzip df.zip
unzip indices.zip
```

### 5. Run the backend (FastAPI)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs will be available at `http://127.0.0.1:8000/docs`.

### 6. Run the frontend (Streamlit)

In `app.py`, `API_BASE` points to the deployed backend by default. For local development, update it to your local API:

```python
API_BASE = "http://127.0.0.1:8000"
```

Then start the app:

```bash
streamlit run app.py
```

---

## ☁️ Deployment

The repo includes a `render.yaml` that defines two Render web services:

- **`movie-rec-api`** — the FastAPI backend (`uvicorn main:app`)
- **`movie-rec-frontend`** — the Streamlit UI (`streamlit run app.py`)

Both can be deployed directly from this repo using [Render's Blueprint feature](https://render.com/docs/blueprint-spec). Remember to set the `TMDB_API_KEY` environment variable manually in the Render dashboard for the API service, since it's marked `sync: false` for security.

---

## 🧠 How the Recommendations Work

1. **Data preparation** — movie metadata (title, overview, genres, etc.) is cleaned and combined into a text "soup" in the notebook.
2. **TF-IDF vectorization** — the text is transformed into a sparse TF-IDF matrix capturing term importance across the corpus.
3. **Similarity computation** — for a selected movie, cosine similarity is computed against all other TF-IDF vectors to rank the most similar titles.
4. **Enrichment** — the top matching titles are looked up on TMDB to attach posters, release dates, and other metadata before being returned to the frontend.
5. **Genre fallback** — a separate genre-based feed (via TMDB's discover endpoint) complements the TF-IDF results with more mainstream, popularity-ranked suggestions.

---

