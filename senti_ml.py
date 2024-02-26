import os
import pandas as pd 
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

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

dataset_folder = "./aclImdb"

def load_dataset():
    data = {'review':[],'sentiment':[]}
    for split in ['train','test']:
        for sentiment in ['pos','neg']:
            folder = os.path.join(dataset_folder,split,sentiment)
            for file_name in os.listdir(folder):
                with open(os.path.join(folder,file_name),'r',encoding="UTF-8") as file:
                    review = file.read()
                    data['review'].append(review)
                    data['sentiment'].append(sentiment)
    return pd.DataFrame(data)

dataset = load_dataset()
dataset['processed_review'] = dataset['review'].apply(preprocess_text)
print(dataset.head())

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("Training set shape:", train_data.shape)
print("Testing set shape:", test_data.shape)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['processed_review'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['processed_review'])

print("Training set TF-IDF shape:", X_train_tfidf.shape)
print("Testing set TF-IDF shape:", X_test_tfidf.shape)

# Define test_labels by extracting sentiment labels from test_data
test_labels = test_data['sentiment']

# applying logistic regression model
logreg_model = LogisticRegression()

logreg_model.fit(X_train_tfidf, train_data['sentiment'])
predicted_labels = logreg_model.predict(X_test_tfidf)

# Evaluate model performance
accuracy = accuracy_score(test_labels, predicted_labels)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(test_labels, predicted_labels))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(test_labels, predicted_labels))