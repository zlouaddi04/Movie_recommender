from src.data import load_and_preprocess
from src.models import build_model
from src.recommender import recommend

# -------------------------
# Load data
# -------------------------
movies = load_and_preprocess()

# -------------------------
# Build model
# -------------------------
matrix, model = build_model(movies)

# -------------------------
# Test recommendation
# -------------------------
movie_name = "Avatar"
results = recommend(movie_name, movies, model, matrix)


print(f"\nRecommendations for {movie_name}:\n")

for title, score in results:
    print(f"{title} (score: {score:.3f})")



