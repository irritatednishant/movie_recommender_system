# Movie Recommender System

A content-based movie recommender implemented as a Jupyter Notebook using TMDB sample datasets. It builds recommendations by combining movie metadata (genres, keywords, cast, crew, and overview) into a textual feature set and finding similar movies using vectorized text and cosine similarity.

## Key idea

This project creates a combined "tags" representation for each movie from metadata fields and converts the tags into numeric vectors using scikit-learn's CountVectorizer. Pairwise cosine similarity between movie vectors is used to find the most similar movies for any given title.

## Datasets

- CSVs/tmdb_5000_movies.csv — movie metadata (genres, overview, title, etc.)
- CSVs/tmdb_5000_credits.csv — cast and crew information
- Extras/movies.pkl — optional preprocessed data

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- nltk (used for stemming)

Example install:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy scikit-learn nltk
```

## How to run

1. Clone the repository:

```bash
git clone https://github.com/irritatednishant/movie_recommender_system.git
cd movie_recommender_system
```

2. Ensure the CSV files listed above are present in the CSVs/ directory.
3. Install dependencies (see Requirements).
4. Open `notebooks/Movie_recommender.ipynb` with Jupyter Notebook or JupyterLab and run the cells in order.

## Example usage

In the notebook, after the similarity matrix is built, call:

```python
recommend('Avatar')
```

This will print the top 5 movies most similar to the provided title according to the content-based model.

## Limitations

- Content-based recommendations rely on metadata and do not use user ratings or collaborative filtering.
- Merging on `title` can be ambiguous when multiple movies share the same title; using a unique movie identifier is more robust.
- Results may be improved by using TF-IDF, sentence embeddings, or refined preprocessing and feature selection.

## Files

- notebooks/Movie_recommender.ipynb — main notebook implementing the pipeline and `recommend()` function
- CSVs/tmdb_5000_movies.csv — TMDB movies dataset
- CSVs/tmdb_5000_credits.csv — TMDB credits dataset
- Extras/movies.pkl — optional preprocessed pickle
