import pandas as pd
import torch
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import streamlit as st
from transformers import BertTokenizer,BertForSequenceClassification,Trainer, TrainingArguments

# Download required NLTK data
nltk.download('stopwords', quiet=True)

# Load stopwords into a set for faster lookup
stop_words = set(stopwords.words('english'))

# Load model and tokenizer
# Note: Make sure to download the model from Google Drive first
# The model folder should be placed in the project directory
try:
    model = BertForSequenceClassification.from_pretrained('./sentiment_model')
    tokenizer = BertTokenizer.from_pretrained('./sentiment_model')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Model not found! Please download the model from Google Drive.\nError: {str(e)}")
    st.info("Download the model folder from: [Google Drive Link]")
    st.stop()

#preprocess data
def cleen_text_steming(text):
    porter=PorterStemmer()
    text=text.lower()
    text=re.sub(r'[^a-zA-Z\s]',' ',text).split()  # Fixed regex pattern
    text=[w for w in text if w not in stop_words]  # Use stop_words set instead of stopwords
    text=[porter.stem(w) for w in text]
    text=' '.join(text)
    return text
#Streamlit UI app
st.title('Sentiment Analysis')

text = st.text_input('Enter your review: ')
button=st.button('Analyze')
if text and button:
    text=cleen_text_steming(text)
    inputs=tokenizer(text,return_tensors='pt',truncation=True,padding=True,max_length=438)
    outputs=model(**inputs)
    pred=torch.argmax(outputs.logits,dim=1)
    
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = label_map[pred.item()]
    
    st.success(f'Prediction: {sentiment}')
else:
    st.write('Please enter your review and click Analyze')

