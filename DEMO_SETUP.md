# Movie Recommender: running the live demo on your laptop

Everything runs on your laptop. You only need internet, because movie posters, details and search come live from TMDB, and the chatbot uses Groq.

**Features:** home feed, search, movie details with **IMDb ratings**, "similar movies" from the TF-IDF model, and a **movie chatbot** (sidebar → 💬 Movie Chatbot) that answers questions like *"Tell me the top 5 horror movies"*.

## 1. Before you start (once)

- **Python 3.10, 3.11 or 3.12.** 3.13 won't work. Get it from https://www.python.org/downloads/ and on Windows tick **"Add Python to PATH"** during install.
- **A free TMDB API key.** Go to https://www.themoviedb.org/settings/api and copy the short 32-character **"API Key"**, not the long "Read Access Token".
- **A free Groq API key** (for the chatbot). Sign in at https://console.groq.com/keys, click **Create API Key**, and copy it. It starts with `gsk_`. Groq shows it only once.

**Unzip it somewhere with a short path**, such as `C:\Movie-Recommender`. Avoid folders nested deep inside OneDrive: Windows can't handle very long file paths, and the install will fail.

## 2. Add your keys

Open the `.env` file in this folder with Notepad or VS Code and replace both placeholders, so it reads:

```
TMDB_API_KEY=your_32_character_key
GROQ_API_KEY=gsk_your_groq_key
```

No spaces and no quotes. On Mac, `.env` is hidden in Finder, so run `open -e .env` in Terminal from this folder instead.

The app works without the Groq key, but then the chatbot page shows "Chatbot is not set up".

## 3. Run it

**Windows:** open **PowerShell** in this folder (Shift + right-click inside the folder → "Open PowerShell window here"). Don't use bash or WSL. Then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1
```

**Mac:** open Terminal in this folder and run:

```bash
bash run_demo.sh
```

The **first run takes 3–5 minutes** because it installs everything. Later runs start in about 10 seconds.

Then open **http://localhost:8501** in your browser.

To stop it on Windows, close the two PowerShell windows that open. On Mac, press Ctrl+C in Terminal.

### Or run it manually (Windows, two terminals)

```powershell
# Terminal 1 - backend (keep open)
cd C:\Movie-Recommender
.\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2 - frontend (keep open)
cd C:\Movie-Recommender
.\venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

If Streamlit asks for an "Email:", just press Enter.

## On demo day

- **Turn on Cloudflare WARP** (the "1.1.1.1" app from https://one.one.one.one). Some networks, including Jio, block TMDB. With WARP on, it works almost everywhere.
- **Start the app 5 minutes early.** Click through the movies you'll show, and ask the chatbot the questions you plan to ask. Everything you've opened once loads instantly afterwards, so don't restart the app.
- **Ask the chatbot at a normal pace.** The free Groq plan allows roughly 5–8 questions per minute. It switches between three AI models automatically when one is busy. If it ever says "busy", wait a minute.
- **Keep the terminal windows open** (minimize them). Closing one stops the app.
- **Don't show `.env` on screen**, because it contains your API keys.
- **Keep a backup:** a 1–2 minute screen recording of the app working.

## Good chatbot questions for the panel

- Tell me the top 5 horror movies
- Only ones from after 2010 *(a follow-up; it remembers the conversation)*
- Best Hindi comedies since 2015
- Movies like Inception *(uses the project's own TF-IDF model)*
- Who directed Interstellar and what's its IMDb rating?
- 3 Tom Hanks movies
- What's trending this week?

Every movie, year and rating in the answers comes from TMDB, IMDb or the TF-IDF model, not from the AI's memory, so it doesn't make things up. Posters appear under each answer. Click **Open** on a poster to see that movie's details page.

## Useful links while the app is running

| What | URL |
|---|---|
| The app | http://localhost:8501 |
| A movie details page | http://localhost:8501/?view=details&id=1771 |
| The chatbot | http://localhost:8501/?view=chat |
| Backend status | http://127.0.0.1:8000/health |
| Backend API docs (nice to show the panel) | http://127.0.0.1:8000/docs |

## If something goes wrong

| Problem | Fix |
|---|---|
| "Put your TMDB API key in .env first" | The key isn't saved in `.env` (step 2). |
| "Python 3.10, 3.11 or 3.12 is required" | Install one of those versions (step 1). |
| "running scripts is disabled" (Windows) | Use the exact command above. The `-ExecutionPolicy Bypass` part fixes it. |
| A "Windows Subsystem for Linux" window appears | Press Esc and close it. You used `bash` on Windows; use PowerShell instead. |
| "TMDB request error: ConnectTimeout" | TMDB is blocked on your network. Turn on WARP or use a different network. |
| Pages show "TMDB error 401" | The TMDB key is wrong. Use the short "API Key", not the Read Access Token. |
| Chatbot: "Chatbot is not set up" | Add `GROQ_API_KEY=...` to `.env`, then restart the app (close both windows and run again). |
| Chatbot: "Groq rejected the API key" | The Groq key is wrong or was deleted. Create a new one at console.groq.com/keys. |
| Chatbot: "The chatbot is busy" | Free-tier limit reached. Wait about a minute and ask again. |
| "Port(s) ... already in use" | The app is already running, so just open http://localhost:8501. To restart it, close the two app windows first. |
| "This folder's path is too long" | Move the `Movie-Recommender` folder somewhere shorter, e.g. `C:\Movie-Recommender`, and run again. |
| "The filename or extension is too long" during install | Same fix: move the folder to a shorter path, delete the `venv` folder inside it, and run again. |

## What's in this folder

| File | What it is |
|---|---|
| `main.py` | FastAPI backend: TF-IDF recommendations, TMDB data, IMDb ratings and the chatbot |
| `app.py` | Streamlit frontend (the website) |
| `df.pkl`, `indices.pkl`, `tfidf.pkl`, `tfidf_matrix.pkl` | Trained recommendation model (45,447 movies) |
| `imdb_ratings.tsv.gz` | IMDb's official ratings dataset (datasets.imdbws.com), for titles with 25+ votes |
| `Movie_Recommender_System.ipynb` | Notebook that built the model |
| `run_demo.ps1` / `run_demo.sh` | One-command start scripts (Windows / Mac) |
