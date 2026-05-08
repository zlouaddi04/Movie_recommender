from src.data import load_and_preprocess
from src.models import build_model
from src.recommender import recommend


# ------------------------------------------------
# Load data
# ------------------------------------------------
movies = load_and_preprocess()

print("Dataset loaded")

# ------------------------------------------------
# Build model
# ------------------------------------------------
matrix, model = build_model(movies)

print("Model built")


# ------------------------------------------------
# Test loop
# ------------------------------------------------
while True:

    movie_name = input(
        "\nEnter movie name: "
    )

    results = recommend(
        movie_name,
        movies,
        model,
        matrix,
        top_n=5
    )

    if len(results) == 0:
        print("Movie not found")
        continue

    print("\nRecommendations:\n")

    for title, score in results:
        print(
            f"{title} -> {score:.3f}"
        )


