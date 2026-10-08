\# 🎬 Movie Recommender System



An AI/ML-based movie recommendation system built using Python, Scikit-learn, FastAPI, Streamlit, and the TMDB API.



\## 🚀 Features



\* Movie recommendations using TF-IDF-based content similarity

\* Movie search using the TMDB API

\* Popular and trending movie listings

\* Genre-based recommendations

\* Movie details

\* FastAPI REST API

\* Interactive Streamlit frontend

\* Environment-based API key management



\## 🛠️ Tech Stack



\* \*\*Python\*\*

\* \*\*FastAPI\*\*

\* \*\*Uvicorn\*\*

\* \*\*Streamlit\*\*

\* \*\*Scikit-learn\*\*

\* \*\*Pandas\*\*

\* \*\*NumPy\*\*

\* \*\*SciPy\*\*

\* \*\*TMDB API\*\*

\* \*\*httpx\*\*

\* \*\*python-dotenv\*\*



\## 🤖 Machine Learning



The recommendation system uses \*\*TF-IDF (Term Frequency-Inverse Document Frequency)\*\* to convert movie information into numerical feature vectors and identify movies with similar content.



The project includes data preprocessing, feature extraction, recommendation generation, API integration, and an interactive frontend.



\## 📡 API Endpoints



```text

GET /health

GET /home?category=popular\&limit=24

GET /tmdb/search?query=...

GET /movie/id/{tmdb\_id}

GET /recommend/tfidf?title=...

GET /recommend/genre?tmdb\_id=...

GET /movie/search?query=...

```



FastAPI documentation:



```text

http://127.0.0.1:8000/docs

```



\## 📁 Project Structure



```text

Movie-Recommender/

│

├── app.py

├── main.py

├── Movie\_Recommender\_System.ipynb

├── df.pkl

├── indices.pkl

├── tfidf.pkl

├── tfidf\_matrix.pkl

├── requirements.txt

├── runtime.txt

├── render.yaml

├── README.md

└── .gitignore

```



\## ⚙️ Installation



Clone the repository:



```bash

git clone https://github.com/kajalbehera344/movie-recommender-ai.git

cd movie-recommender-ai

```



Create a virtual environment:



```bash

python -m venv venv

```



For Windows PowerShell:



```powershell

.\\venv\\Scripts\\Activate.ps1

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\## 🔑 Environment Variables



Create a `.env` file:



```text

TMDB\_API\_KEY=your\_api\_key\_here

```



Do \*\*not\*\* upload your actual API key to GitHub.



\## ▶️ Run the Backend



```bash

uvicorn main:app --reload --host 0.0.0.0 --port 8000

```



\## 🖥️ Run the Frontend



Open another terminal and run:



```bash

streamlit run app.py

```



\## 📌 Project Highlights



\* Built an end-to-end AI/ML movie recommendation application.

\* Implemented TF-IDF-based recommendation logic.

\* Developed REST APIs using FastAPI.

\* Integrated TMDB API for movie data and search.

\* Created an interactive Streamlit interface.

\* Used Pandas, NumPy, Scikit-learn, and SciPy for data processing and machine learning.



\## 👩‍💻 Author



\*\*Kajal Behera\*\*



GitHub: https://github.com/kajalbehera344



Project Repository: https://github.com/kajalbehera344/movie-recommender-ai



