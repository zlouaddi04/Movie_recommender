import numpy as np

def recommend(movie_title, movies, model, matrix, top_n=5):
    # -------------------------
    # Find movie index
    # -------------------------
    idx = movies[movies['title'] == movie_title].index[0]

    # -------------------------
    # Get neighbors
    # -------------------------
    distances, indices = model.kneighbors(matrix[idx], n_neighbors=top_n + 1)

    results = []

    for i in range(1, len(indices[0])):
        movie_idx = indices[0][i]

        similarity = 1 - distances[0][i]
        vote = movies.iloc[movie_idx]['vote_average']
        success = movies.iloc[movie_idx]['success']

        # -------------------------
        # Final score (hybrid)
        # -------------------------
        final_score = (
            0.7 * similarity +
            0.2 * vote +
            0.1 * success
        )

        results.append((movies.iloc[movie_idx]['title'], final_score))

    # -------------------------
    # Sort by score
    # -------------------------
    results = sorted(results, key=lambda x: x[1], reverse=True)

    return results