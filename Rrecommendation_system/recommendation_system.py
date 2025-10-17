# Simple Movie Recommendation System (Content-Based)
# By Sowndar B 😊

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Sample dataset
data = {
    'movie_id': [1, 2, 3, 4, 5, 6],
    'title': ['Inception', 'Interstellar', 'The Dark Knight', 'Avatar', 'Titanic', 'The Matrix'],
    'genre': ['Sci-Fi Thriller', 'Sci-Fi Adventure', 'Action Crime', 'Sci-Fi Fantasy', 'Romance Drama', 'Sci-Fi Action']
}

# Create a DataFrame
movies = pd.DataFrame(data)

# Combine important features
movies['features'] = movies['title'] + " " + movies['genre']

# Convert text to feature vectors
vectorizer = CountVectorizer()
feature_vectors = vectorizer.fit_transform(movies['features'])

# Compute cosine similarity
similarity = cosine_similarity(feature_vectors)

# Recommendation function
def recommend(movie_title):
    if movie_title not in movies['title'].values:
        print("Sorry, movie not found in the database.")
        return

    movie_index = movies[movies['title'] == movie_title].index[0]
    scores = list(enumerate(similarity[movie_index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]  # top 3 similar movies

    print(f"\nMovies similar to '{movie_title}':")
    for i, score in sorted_scores:
        print(f"👉 {movies.iloc[i]['title']} ({movies.iloc[i]['genre']}) - Similarity: {score:.2f}")

# Example usage
print("🎬 Welcome to Movie Recommendation System 🎬")
print("\nAvailable movies:")
print(movies['title'].to_string(index=False))

movie_name = input("\nEnter a movie you like: ")
recommend(movie_name)

