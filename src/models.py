from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def build_model(movies):

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )

    matrix = tfidf.fit_transform(
        movies["content"]
    )

    knn_model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )

    knn_model.fit(matrix)

    return matrix, knn_model