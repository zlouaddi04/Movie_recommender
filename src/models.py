from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def build_model(movies):
    # -------------------------
    # TF-IDF on content
    # -------------------------
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    matrix = tfidf.fit_transform(movies["content"])

    # -------------------------
    # KNN model
    # -------------------------
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(matrix)

    return matrix, knn