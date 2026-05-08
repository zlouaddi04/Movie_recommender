def recommend(
    movie_title,
    movies,
    model,
    matrix,
    top_n=5
):

    # ------------------------------------------------
    # Case-insensitive search
    # ------------------------------------------------
    movie_title = movie_title.lower().strip()

    matched = movies[
        movies["search_title"] == movie_title
    ]

    if matched.empty:
        return []

    idx = matched.index[0]

    # ------------------------------------------------
    # Current movie genres
    # ------------------------------------------------
    current_genres = set(
        movies.iloc[idx]["genres"].split()
    )

    # ------------------------------------------------
    # Get neighbors
    # ------------------------------------------------
    distances, indices = model.kneighbors(
        matrix[idx],
        n_neighbors=top_n * 4
    )

    results = []

    # skip itself
    for i in range(1, len(indices[0])):

        movie_idx = indices[0][i]

        similarity = (
            1 - distances[0][i]
        )

        vote = movies.iloc[movie_idx][
            "vote_average"
        ]

        popularity = movies.iloc[movie_idx][
            "popularity"
        ]

        success = movies.iloc[movie_idx][
            "success"
        ]

        # ------------------------------------------------
        # Genre overlap
        # ------------------------------------------------
        candidate_genres = set(
            movies.iloc[movie_idx]["genres"].split()
        )

        shared_genres = current_genres.intersection(
            candidate_genres
        )

        # reject unrelated genres
        if len(shared_genres) == 0:
            continue

        genre_score = (
            len(shared_genres) /
            len(current_genres)
        )

        # ------------------------------------------------
        # Hybrid score
        # ------------------------------------------------
        final_score = (
            0.5 * similarity +
            0.2 * vote +
            0.15 * popularity +
            0.1 * success +
            0.05 * genre_score
        )

        results.append(
            (
                movies.iloc[movie_idx]["title"],
                final_score
            )
        )

    # ------------------------------------------------
    # Sort descending
    # ------------------------------------------------
    results = sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )

    return results[:top_n]