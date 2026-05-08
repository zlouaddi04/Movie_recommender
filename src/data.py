import pandas as pd
import ast
import numpy as np


# ------------------------------------------------
# Extract genre names from JSON-like strings
# ------------------------------------------------
def extract_names(obj):
    try:
        obj = ast.literal_eval(obj)
        return [item["name"] for item in obj]
    except:
        return []


# ------------------------------------------------
# Revenue / budget success metric
# ------------------------------------------------
def compute_success(row):

    budget = row["budget"]
    revenue = row["revenue"]

    if budget == 0 or revenue == 0:
        return 0

    return revenue / budget


# ------------------------------------------------
# Main preprocessing
# ------------------------------------------------
def load_and_preprocess():

    movies = pd.read_csv(
        "Data/movies_metadata.csv",
        low_memory=False
    )

    # ------------------------------------------------
    # Keep useful columns
    # ------------------------------------------------
    movies = movies[
        [
            "title",
            "overview",
            "genres",
            "vote_average",
            "vote_count",
            "popularity",
            "budget",
            "revenue",
            "poster_path",
            "release_date",
        ]
    ]

    # ------------------------------------------------
    # Missing values
    # ------------------------------------------------
    movies = movies.fillna("")

    # ------------------------------------------------
    # Convert numeric columns
    # ------------------------------------------------
    movies["budget"] = pd.to_numeric(
        movies["budget"],
        errors="coerce"
    ).fillna(0)

    movies["revenue"] = pd.to_numeric(
        movies["revenue"],
        errors="coerce"
    ).fillna(0)

    movies["popularity"] = pd.to_numeric(
        movies["popularity"],
        errors="coerce"
    ).fillna(0)

    movies["vote_average"] = pd.to_numeric(
        movies["vote_average"],
        errors="coerce"
    ).fillna(0)

    movies["vote_count"] = pd.to_numeric(
        movies["vote_count"],
        errors="coerce"
    ).fillna(0)

    # ------------------------------------------------
    # Extract genres
    # ------------------------------------------------
    movies["genres"] = movies["genres"].apply(extract_names)

    movies["genres"] = movies["genres"].apply(
        lambda x: " ".join([g.lower() for g in x])
    )

    # ------------------------------------------------
    # Clean overview
    # ------------------------------------------------
    movies["overview"] = movies["overview"].astype(str).str.lower()

    # ------------------------------------------------
    # Weighted content
    # ------------------------------------------------
    movies["content"] = (
        movies["overview"] + " " +
        movies["genres"] + " " +
        movies["genres"]
    )

    # ------------------------------------------------
    # Case-insensitive search
    # ------------------------------------------------
    movies["search_title"] = (
        movies["title"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # ------------------------------------------------
    # SUCCESS metric
    # ------------------------------------------------
    movies["success"] = movies.apply(
        compute_success,
        axis=1
    )

    # log scaling
    movies["success"] = np.log1p(movies["success"])

    # normalize
    if movies["success"].max() > 0:
        movies["success"] = (
            (movies["success"] - movies["success"].min()) /
            (movies["success"].max() - movies["success"].min())
        )

    # ------------------------------------------------
    # Normalize popularity
    # ------------------------------------------------
    if movies["popularity"].max() > 0:
        movies["popularity"] = (
            (movies["popularity"] - movies["popularity"].min()) /
            (movies["popularity"].max() - movies["popularity"].min())
        )

    # ------------------------------------------------
    # Normalize ratings
    # ------------------------------------------------
    movies["vote_average"] = (
        movies["vote_average"] / 10.0
    )

    # ------------------------------------------------
    # Remove empty movies
    # ------------------------------------------------
    movies = movies[
        movies["content"].str.strip() != ""
    ]

    movies = movies.reset_index(drop=True)

    return movies