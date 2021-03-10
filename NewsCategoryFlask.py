
from flask import Flask, jsonify, request
import joblib
import pandas as pd
import requests
import numpy as np
import scipy
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('wordnet')
nltk.download('punkt')
nltk.download("stopwords")

app = Flask(__name__)

tfidfvectorizer = joblib.load('models/X_train_news_tfidf.pkl')
LRmodel = joblib.load('models/LR_model.pkl')
stop_words = stopwords.words('english')

def text_preprocess(text):
      lemmatizer = WordNetLemmatizer()
      string = ''
      sentence = text
      sentence = re.sub(r'[^\w\s]',' ',sentence)
      sentence = re.sub(r"\s+"," ", sentence, flags = re.I)#removing extra space
      sentence = re.sub(r"\d", " ", sentence)#removing digits
      sentence = re.sub(r"\s+[a-zA-Z]\s+", " ", sentence)#removing single characters
      sentence = re.sub(r"[,@\'?\.$%_]", "", sentence, flags=re.I)#removing multiple characters

      sentence = sentence.lower()

      words = nltk.word_tokenize(sentence)
      for word in words:
        # if the word is a not a stop word then retain that word from the data
            if not word in stop_words:
                string += str(lemmatizer.lemmatize(word)) + " "
      return string

def scorer(text):
    text = text_preprocess(text)
    vector_text =  tfidfvectorizer.transform([text])
    score = LRmodel.predict(vector_text)
    return score

@app.route('/predict', methods=['POST'])
def predict_fn():
    text = request.get_json()['text']
    predictions = scorer(text)
    predictions =predictions.argmax(axis=1)[0]
    #return jsonify({'predictions ': str(predictions), 'Category ': target.get(predictions)})
    return jsonify({'predictions ': str(predictions), 'Category ': target.get(predictions)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000')