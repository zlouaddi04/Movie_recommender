import pandas as pd
import ast
import numpy as np

# -------------------------
# Helper: extract names
# -------------------------
def extract_names(obj):
    try:
        obj = ast.literal_eval(obj)
        return [item['name'] for item in obj]
    except:
        return []


# -------------------------
# Helper: success score
# -------------------------
def compute_success(row):
    budget = row['budget']
    revenue = row['revenue']

    if budget == 0 or revenue == 0:
        return 0

    return revenue / budget


# -------------------------
# Main preprocessing
# -------------------------
def load_and_preprocess():
    movies = pd.read_csv("Data/tmdb_5000_movies.csv")

    # -------------------------
    # Keep useful columns
    # -------------------------
    movies = movies[
        [
            "title",
            "overview",
            "genres",
            "keywords",
            "original_language",
            "vote_average",
            "budget",
            "revenue"
        ]
    ]

    # -------------------------
    # Handle missing values
    # -------------------------
    movies = movies.fillna("")

    # -------------------------
    # Extract JSON fields
    # -------------------------
    movies["genres"] = movies["genres"].apply(extract_names)
    movies["keywords"] = movies["keywords"].apply(extract_names)

    # -------------------------
    # Convert lists → strings
    # -------------------------
    movies["genres"] = movies["genres"].apply(lambda x: " ".join(x))
    movies["keywords"] = movies["keywords"].apply(lambda x: " ".join(x))

    # -------------------------
    # Lowercase text
    # -------------------------
    movies["overview"] = movies["overview"].str.lower()
    movies["genres"] = movies["genres"].str.lower()
    movies["keywords"] = movies["keywords"].str.lower()

    # -------------------------
    # Create content column
    # -------------------------
    movies["content"] = (
        movies["overview"] + " " +
        movies["genres"] + " " +
        movies["keywords"]
    )

    # -------------------------
    # Compute SUCCESS metric
    # -------------------------
    movies["success"] = movies.apply(compute_success, axis=1)

    # Log scaling (reduce extreme values)
    movies["success"] = np.log1p(movies["success"])

    # Normalize (0 → 1)
    if movies["success"].max() > 0:
        movies["success"] = (
            (movies["success"] - movies["success"].min()) /
            (movies["success"].max() - movies["success"].min())
        )

    # -------------------------
    # Clean vote_average (normalize too)
    # -------------------------
    movies["vote_average"] = movies["vote_average"] / 10.0

    return movies

