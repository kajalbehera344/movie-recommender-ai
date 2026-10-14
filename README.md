🎬 Movie Recommender System

A movie recommendation web application built with Python, Machine Learning, FastAPI, Streamlit, and The Movie Database (TMDB) API. Discover popular movies, explore genres, search for films, and get similar movie recommendations using content-based filtering.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 Features

🎥 Movie Discovery  
Explore popular and trending movies.

🤖 Machine Learning Recommendations  
Discover similar movies using TF-IDF vectorization and cosine similarity.

🔍 Movie Search  
Search for movies using the TMDB API.

🎭 Genre-Based Recommendations  
Find movies based on their genres.

📄 Movie Details  
View movie information and available metadata.

⚡ FastAPI Backend  
Provides REST API endpoints for movie search and recommendations.

🖥️ Interactive Frontend  
Browse and discover movies through a Streamlit interface.

🔐 Secure API Configuration  
Store API keys in environment variables instead of source code.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🛠️ Tech Stack

Frontend

- Streamlit
- Python

Backend

- FastAPI
- Uvicorn
- HTTPX

Machine Learning and Data Processing

- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Pandas
- NumPy
- SciPy

External API

- TMDB API
- Groq API (for the AI assistant, when implemented)

Development Tools

- Git
- GitHub
- VS Code
- Python Virtual Environment

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤖 Machine Learning Approach

The recommendation system uses content-based filtering to identify movies with similar characteristics.

1. Data Preparation — Movie information is loaded and processed using Pandas.
2. Feature Extraction — TF-IDF converts textual movie features into numerical vectors.
3. Similarity Calculation — Similarity scores identify movies with related content.
4. Recommendation Generation — Movies with the highest similarity scores are returned as recommendations.

This approach helps users discover movies related to their interests without requiring ratings from other users.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📡 API Endpoints

The FastAPI backend provides endpoints for movie discovery, search, and recommendations.

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Check API status |
| GET | /home?category=popular&limit=24 | Retrieve movies for the home page |
| GET | /tmdb/search?query=Inception | Search movies using TMDB |
| GET | /movie/id/{tmdb_id} | Retrieve movie details |
| GET | /recommend/tfidf?title=Inception | Get content-based recommendations |
| GET | /recommend/genre?tmdb_id=27205 | Get genre-based recommendations |
| GET | /movie/search?query=Inception | Search movies through the application |

Endpoint availability and parameter names depend on the implementation in `main.py`.

Interactive API documentation:  
http://127.0.0.1:8000/docs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 Project Structure

Movie-Recommender/

- app.py — Streamlit frontend
- main.py — FastAPI backend
- Movie_Recommender_System.ipynb — Notebook for data analysis and model development
- df.pkl — Saved movie dataset
- indices.pkl — Saved movie index mapping
- tfidf.pkl — Saved TF-IDF vectorizer
- tfidf_matrix.pkl — Saved TF-IDF feature matrix
- requirements.txt — Python dependencies
- runtime.txt — Python runtime configuration
- render.yaml — Deployment configuration, if configured
- .env.example — Environment variable template
- .gitignore — Files excluded from Git

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚙️ Installation and Setup

1. Clone the Repository

`git clone https://github.com/kajalbehera344/movie-recommender-ai.git`

`cd movie-recommender-ai`

Make sure you enter the directory containing `main.py` and `app.py`.

2. Create a Virtual Environment

On Windows, run:

`python -m venv venv`

`.\venv\Scripts\Activate.ps1`

3. Install Dependencies

`python -m pip install -r requirements.txt`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔑 Environment Variables

The application uses environment variables to keep API credentials separate from the source code.

1. Create a file named `.env` in the project root, next to `main.py`.

2. Add the following variables:

`TMDB_API_KEY=your_tmdb_api_key`

`GROQ_API_KEY=your_groq_api_key`

The Groq key is needed only if your application code uses Groq. Adding the key alone does not enable AI functionality.

3. Load Environment Variables

Install `python-dotenv` if required:

`python -m pip install python-dotenv`

In the relevant Python file, load the variables:

`import os`

`from dotenv import load_dotenv`

`load_dotenv()`

`TMDB_API_KEY = os.getenv("TMDB_API_KEY")`

`GROQ_API_KEY = os.getenv("GROQ_API_KEY")`

4. Protect Your API Keys

Add these rules to `.gitignore`:

` .env`

`.env.*`

`!.env.example`

`venv/`

`.venv/`

`__pycache__/`

`*.py[cod]`

Never upload your real `.env` file or API keys to GitHub.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▶️ Run the Application

1. Start the FastAPI Backend

Run this command in the project directory:

`uvicorn main:app --reload --host 127.0.0.1 --port 8000`

Backend URL:  
http://127.0.0.1:8000

Interactive API Documentation:  
http://127.0.0.1:8000/docs

Keep this terminal running.

2. Start the Streamlit Frontend

Open a second terminal in the project directory and run:

`streamlit run app.py`

Open the local URL displayed in the terminal, usually:

http://localhost:8501

Make sure the frontend is configured to communicate with the backend correctly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌐 Deployment

The application can be deployed using compatible hosting services.

Backend — Render or another Python-compatible hosting service.

Frontend — Streamlit Community Cloud or another compatible hosting service.

Movie Data — TMDB API.

AI Features — Groq API, if implemented in the application code.

Before deployment, configure environment variables in the hosting provider's dashboard, update the frontend to use the deployed backend URL, and ensure all required model files and dependencies are available.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📸 Screenshots

Add actual screenshots of your application inside a folder named `screenshots`.

Home Page  
`screenshots/home.png`

Movie Recommendations  
`screenshots/recommendations.png`

Movie Details  
`screenshots/movie-details.png`

Use clear, original screenshots and make sure the image files exist before adding them to the README.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 Project Highlights

- Combines machine learning with an interactive movie discovery interface.
- Uses content-based filtering for similar-movie recommendations.
- Separates frontend and backend responsibilities.
- Integrates an external movie database API.
- Demonstrates Python development, REST API design, data processing, and model integration.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔮 Future Improvements

- Add a conversational AI movie assistant using Groq.
- Support natural-language movie searches.
- Add user accounts and personalized watchlists.
- Improve recommendations using user preferences and feedback.
- Add automated testing and enhanced error handling.

These are potential improvements and should not be considered implemented until they are added to the project.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👩‍💻 Author

Kajal Behera

GitHub:  
https://github.com/kajalbehera344

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This project was developed for learning and demonstrating machine learning, Python application development, and API integration.

Movie data and related metadata are provided by TMDB. This project is not endorsed or certified by TMDB.