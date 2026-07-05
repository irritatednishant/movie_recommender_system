# Movie Recommender System

A simple content-based movie recommender implemented as a Jupyter Notebook using the TMDB (The Movie Database) sample datasets. It builds a recommendation engine by combining movie metadata (genres, keywords, cast, crew, and overview) into a single textual "tag" per movie, vectorizing the tags, and using cosine similarity to find similar movies.

## Key idea
This project is a content-based recommender: for any given movie title it finds other movies with similar metadata. It does NOT use explicit user ratings or collaborative filtering — recommendations are based on movie attributes/content only.

## Datasets
- CSVs: The repository includes two TMDB CSV files under `CSVs/`:
  - `tmdb_5000_movies.csv` — movie metadata (genres, overview, title, etc.)
  - `tmdb_5000_credits.csv` — cast and crew information
- Extras:
  - `Extras/movies.pkl` — a preprocessed pickle (likely intermediate or saved results)

If you don't have the CSVs locally, download the TMDB sample datasets (commonly available on Kaggle or other public sources) and place them in the `CSVs/` directory with the filenames above.

## What the notebook does (notebooks/Movie_recommender.ipynb)
1. Loads `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` and merges them on the `title` column.
2. Selects and keeps these fields: `movie_id`, `genres`, `keywords`, `overview`, `title`, `cast`, `crew`.
3. Drops rows with missing values and checks for duplicates.
4. Parses JSON-like string columns (`genres`, `keywords`, `cast`, `crew`) into lists of names using `ast.literal_eval` and helper functions.
5. Creates a combined textual `tags` column per movie by concatenating (and lightly processing) genres, keywords, cast, crew and overview text.
6. Applies basic text preprocessing (tokenization/stemming) on the `tags`.
7. Uses scikit-learn's `CountVectorizer(max_features=5000, stop_words='english')` to convert tags into numeric vectors (`(n_movies, 5000)` shape).
8. Computes pairwise cosine similarity between movie vectors with `cosine_similarity`.
9. Implements `recommend(movie_title)` which finds the top 5 most similar movies (by cosine similarity) and prints their titles.

## Requirements
- Python 3.8+ recommended
- The notebook uses these libraries (install with pip):
  - pandas
  - numpy
  - scikit-learn
  - nltk (for stemming; the notebook uses PorterStemmer)

Example install commands:

```bash
python -m venv .venv
source .venv/bin/activate    # on Windows use: .venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy scikit-learn nltk
```

If you prefer, create `requirements.txt` with the above and run `pip install -r requirements.txt`.

## How to run
1. Clone the repository:

```bash
git clone https://github.com/irritatednishant/movie_recommender_system.git
cd movie_recommender_system
```

2. Ensure the CSVs are present in `CSVs/` (the repo already contains `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`).
3. Install dependencies (see Requirements).
4. Open the notebook `notebooks/Movie_recommender.ipynb` with Jupyter Notebook / JupyterLab and run the cells in order.

Alternatively, if you want a quick test without running the whole notebook, the repository includes `Extras/movies.pkl` which may contain preprocessed data — you can inspect it in Python with `pd.read_pickle('Extras/movies.pkl')`.

## Example usage
Open the notebook and after the similarity matrix is built, call in a cell:

```python
recommend('Avatar')
```

The function prints 5 movie titles similar to the input title based on the combined metadata.

## Limitations and notes
- This is a content-based approach (no user profiles or rating data). It finds movies that are similar by metadata, not necessarily what a specific user prefers.
- The current pipeline uses simple stemming/tokenization and a CountVectorizer; results can be improved by:
  - Using TF-IDF instead of raw counts
  - Using more advanced text embeddings (word2vec, BERT sentence embeddings)
  - Improving cast/crew parsing (e.g., selecting top-N cast members or filtering by role)
  - Handling titles collisions when multiple movies share the same title (merging on `title` can be ambiguous)
- The notebook merges datasets on the title, which can cause mismatches if titles differ (typos, year suffixes, duplicates). A more robust merge would use a unique movie identifier if available.

## Files
- notebooks/Movie_recommender.ipynb — main notebook implementing the pipeline and `recommend()` function
- CSVs/tmdb_5000_movies.csv — TMDB movies dataset
- CSVs/tmdb_5000_credits.csv — TMDB credits dataset
- Extras/movies.pkl — optional preprocessed pickle

## Next steps / Ideas
- Convert the notebook into a small Flask/FastAPI service that exposes recommendations via an API endpoint.
- Replace CountVectorizer + cosine similarity with sentence embeddings (e.g., SentenceTransformers) to capture better semantic similarity.
- Add unit tests and a small CLI script to query recommendations from the command line.

---

If you'd like, I can: (a) add a requirements.txt, (b) extract the notebook into a runnable script, or (c) convert this README into a shorter project summary. Tell me which you'd prefer and I'll add it.