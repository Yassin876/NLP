# Import the libraries
import streamlit as st
import pandas as pd
import re 
import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import BertTokenizer,BertForSequenceClassification,Trainer, TrainingArguments,EarlyStoppingCallback


#download the stopwords
nltk.download('stopwords')
stopwords=set(stopwords.words('english'))
porter=PorterStemmer()


#load the model and tokenizer
model_path = "fake_news_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)



# Clean the text and stem the text
def cleen_text_and_steming(text):
  text = text.lower()
  text = re.sub(r'\S+@\S+', '', text)
  text = re.sub(r'\d+', '', text)
  text = re.sub(r'http\S+|www\S+', '', text)
  text = re.sub(r'<.*?>', '', text).split()
  text = [w for w in text if w not in stopwords]
  text = [porter.stem(w) for w in text ]
  text = ' '.join(text)
  return text


# Predict the news
def predict_news(news):
  news = cleen_text_and_steming(news)
  news = tokenizer(news,return_tensors='pt',padding=True,truncation=True)
  news = {k:v.to(model.device) for k,v in news.items()}
  with torch.no_grad():
    output = model(**news)
    output = torch.softmax(output.logits,dim=1)
    return output.argmax(dim=1).item()


# Streamlit app UI
st.title("Fake News Detection")
news = st.text_input("Enter the news")
button = st.button("Detect")
if news and button:
  result = predict_news(news)
  if result == 1:
    st.write("The news is real")
  elif result == 0:
    st.write("The news is fake")
  else:
    st.write("The news is not detected")

