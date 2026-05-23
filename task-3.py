# ==========================================
# SENTIMENT ANALYSIS USING IMDB DATASET
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("sentiment_data.csv")

print("Dataset Loaded Successfully\n")

print(df.head())


# ==========================================
# CONVERT LABELS
# positive = 1
# negative = 0
# ==========================================

df['sentiment'] = df['sentiment'].map({
    'positive': 1,
    'negative': 0
})


# ==========================================
# INPUT AND OUTPUT
# ==========================================

X = df['review']
y = df['sentiment']


# ==========================================
# SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words='english',
    lowercase=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


# ==========================================
# TRAIN MODEL
# ==========================================

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

print("\nModel Trained Successfully")


# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# ==========================================
# CUSTOM REVIEW TESTING
# ==========================================

while True:

    review = input("\nEnter Review: ")

    review_tfidf = vectorizer.transform([review])

    prediction = model.predict(review_tfidf)

    if prediction[0] == 1:
        print("Sentiment: POSITIVE 😊")
    else:
        print("Sentiment: NEGATIVE 😠")

    choice = input("\nTest another review? (yes/no): ")

    if choice.lower() != 'yes':
        break


print("\nProject Finished")