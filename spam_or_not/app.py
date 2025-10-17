import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.stem import PorterStemmer
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,accuracy_score
import re
from nltk.corpus import stopwords

#load the model
model=joblib.load('the model.pkl')
#load vectorizer 
vectorizer=joblib.load('Vectorizer.pkl')
#load the data
data=pd.read_csv('spam_or_not_spam.csv')
#load the stopwords
stop_words=set(stopwords.words('english'))
#load the stemmer
stemmer=PorterStemmer()
#load the tfidf vectorizer
tfidf=TfidfVectorizer()

# Streamlit UI 
st.title("Spam or Not Spam Classifier")
st.write("inter the email")

user_input = st.text_area("Email Text", height=200)

# clean the user input
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).split()
    text = [w for w in text if w not in stop_words]
    return text

def apply_stemming(text_list):
    text_list = [stemmer.stem(w) for w in text_list]
    return ' '.join(text_list)

if st.button("predict"):
    if user_input.strip() == "":
        st.warning("please inter the email")
    else:
        # apply the clean the user input
        cleaned = clean_text(user_input)
        stemmed = apply_stemming(cleaned)
        # transform the text to vector
        X_vec = vectorizer.transform([stemmed])
        # predict
        pred = model.predict(X_vec)[0]
        label = "Spam " if pred == 1 else "Not Spam "
        st.success(f"the result: {label}")
