# Sentiment Analysis App

A machine learning application that analyzes the sentiment of text reviews using BERT (Bidirectional Encoder Representations from Transformers) model. The app can classify text into three sentiment categories: Negative, Neutral, and Positive.

## Features

- **BERT-based Sentiment Analysis**: Uses a fine-tuned BERT model for accurate sentiment classification
- **Text Preprocessing**: Includes text cleaning, stemming, and stopword removal
- **Interactive Web Interface**: Built with Streamlit for easy-to-use web interface
- **Three-Class Classification**: 
  - Negative (ratings 1-2 stars)
  - Neutral (rating 3 stars)
  - Positive (ratings 4-5 stars)

## Dataset

The model was trained on Disneyland reviews dataset containing:
- 42,656 reviews from different Disneyland locations
- Reviews from Hong Kong, California, and Paris Disneyland
- Balanced dataset with downsampling for better model performance

## Model Architecture

- **Base Model**: BERT-base-uncased
- **Task**: Sequence Classification
- **Classes**: 3 (Negative, Neutral, Positive)
- **Training**: Fine-tuned on review data with 3 epochs
- **Max Length**: 438 tokens (based on average review length)

## Model Download

**Important**: The trained sentiment analysis model is hosted on Google Drive due to its large size (~440MB).

### Download the Model

1. Download the model files from Google Drive:
   - [Download Model Folder](https://drive.google.com/drive/folders/YOUR_MODEL_FOLDER_LINK)
   - Extract the downloaded folder and place it in your project directory
   - Make sure the folder is named `sentiment_model`

2. Alternative: Use the model directly from Google Drive (see Usage section)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sentiment_analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (done automatically in the app):
```python
import nltk
nltk.download('stopwords')
```

4. Download the model from Google Drive (see Model Download section above)

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Using the Model Programmatically

**Option 1: Local Model (after downloading from Drive)**
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer from local folder
model = BertForSequenceClassification.from_pretrained('./sentiment_model')
tokenizer = BertTokenizer.from_pretrained('./sentiment_model')

# Example prediction
text = "I love this place!"
inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=438)
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1)

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
sentiment = label_map[pred.item()]
print(f"Sentiment: {sentiment}")
```

**Option 2: Direct from Google Drive (requires gdown)**
```bash
pip install gdown
```
```python
import gdown
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Download and load model directly from Google Drive
folder_url = "https://drive.google.com/drive/folders/YOUR_MODEL_FOLDER_LINK"
gdown.download_folder(folder_url, quiet=True, use_cookies=False)

model = BertForSequenceClassification.from_pretrained('./sentiment_model')
tokenizer = BertTokenizer.from_pretrained('./sentiment_model')
```

## Project Structure

```
sentiment_analysis/
├── app.py                          # Streamlit web application
├── sentment_analysis.ipynb         # Jupyter notebook with model training
├── sentiment_model/                # Downloaded model folder (from Google Drive)
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── DisneylandReviews.csv           # Training dataset
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

**Note**: The `sentiment_model/` folder is not included in the repository due to its large size. Please download it from Google Drive using the link provided in the Model Download section.

## Model Training Process

The model training process included:

1. **Data Loading**: Loaded Disneyland reviews dataset
2. **Data Exploration**: Analyzed dataset structure and distribution
3. **Text Preprocessing**: 
   - Converted to lowercase
   - Removed punctuation and special characters
   - Removed stopwords
   - Applied stemming using Porter Stemmer
4. **Label Mapping**: Converted 5-star ratings to 3 sentiment classes
5. **Data Balancing**: Applied downsampling to balance classes
6. **Train-Test Split**: 80% training, 20% testing
7. **Tokenization**: Used BERT tokenizer with max_length=438
8. **Model Training**: Fine-tuned BERT with 3 epochs
9. **Model Saving**: Saved trained model and tokenizer

## Performance

- **Training Loss**: 0.66 (after 3 epochs)
- **Evaluation Loss**: 0.76
- **Model Size**: ~440MB
- **Inference Speed**: Fast on GPU, acceptable on CPU

## Text Preprocessing

The app includes comprehensive text preprocessing:

```python
def clean_text_stemming(text):
    porter = PorterStemmer()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).split()
    text = [w for w in text if w not in stop_words]
    text = [porter.stem(w) for w in text]
    text = ' '.join(text)
    return text
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Streamlit
- NLTK
- Pandas
- Scikit-learn

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the application
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Hugging Face for the transformers library
- Google for the BERT model
- Disneyland reviews dataset contributors
- Streamlit for the web framework

## Troubleshooting

### Common Issues

1. **Model not found**: 
   - Make sure you've downloaded the model from Google Drive
   - Ensure the `sentiment_model` folder exists with all required files
   - Check that the folder name is exactly `sentiment_model`
2. **Google Drive download issues**: 
   - Make sure you have access to the Google Drive link
   - Try downloading individual files if the folder download fails
   - Use gdown library for automated downloads
3. **NLTK errors**: Run `nltk.download('stopwords')` if you encounter stopwords issues
4. **Memory issues**: Reduce batch size or use a machine with more RAM
5. **CUDA errors**: The model will automatically fall back to CPU if GPU is not available

### Performance Tips

- Use GPU for faster inference
- Reduce max_length if processing very long texts
- Batch multiple predictions together for efficiency
