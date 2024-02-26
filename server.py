from flask import Flask, request, jsonify
import joblib
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
model = joblib.load('sentiment_analysis_model.pkl')

app = Flask(__name__)

# Load the pre-fitted TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Lowercasing
    tokens = [word.lower() for word in tokens]  

    # Removing punctuation
    tokens = [word for word in tokens if word.isalnum()]

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)  

    return preprocessed_text

@app.route('/api/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    text = data['text']

    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Convert preprocessed text into a TF-IDF vector
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])

    # Use the model to make a prediction
    predicted_sentiment = model.predict(text_tfidf)[0]

    # Return the predicted sentiment as JSON
    return jsonify({'sentiment': predicted_sentiment})

if __name__ == '__main__':
    app.run(debug=True)
