import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load your data
data = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'emotion'])

# Features and labels
X = data['text']
y = data['emotion']

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train classifier
model = LogisticRegression(max_iter=200)
model.fit(X_vectorized, y)

# Save model and vectorizer
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('emotional_model.pkl', 'wb'))

print("Training complete. vectorizer.pkl and emotional_model.pkl saved.")
