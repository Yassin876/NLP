# Fake News Detection System

A machine learning-based fake news detection system using BERT (Bidirectional Encoder Representations from Transformers) with LoRA (Low-Rank Adaptation) fine-tuning. The system can classify news articles as either "Real" or "Fake" with high accuracy.

## Features

- **BERT-based Classification**: Uses pre-trained BERT model fine-tuned for fake news detection
- **LoRA Fine-tuning**: Efficient fine-tuning using Low-Rank Adaptation technique
- **Text Preprocessing**: Comprehensive text cleaning including stopword removal, stemming, and regex-based cleaning
- **Web Interface**: Streamlit-based user-friendly web application
- **High Accuracy**: Achieves 99.8% accuracy on test dataset

## Project Structure

```
facknews/
├── app.py                          # Streamlit web application
├── fack_news.ipynb                 # Jupyter notebook for model training
├── fake_news_model/                # Trained model directory
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer_config.json
│   └── vocab.txt
├── facknews_data/                  # Dataset directory
│   ├── Fake.csv                    # Fake news dataset
│   └── True.csv                    # True news dataset
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Dataset

The model is trained on a dataset containing:
- **Fake News**: 23,481 articles labeled as fake
- **True News**: 21,417 articles labeled as true
- **Total**: 44,898 news articles

The dataset includes news articles with titles and text content from various sources.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd facknews
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**:
   The application will automatically download required NLTK stopwords on first run.

## Usage

### Running the Web Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Enter news text** in the input field and click "Detect" to get the classification result

### Model Training

To retrain the model or understand the training process:

1. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook fack_news.ipynb
   ```

2. **Follow the notebook cells** to:
   - Load and preprocess the dataset
   - Fine-tune the BERT model with LoRA
   - Evaluate model performance
   - Save the trained model

## Model Performance

The trained model achieves the following performance metrics on the test dataset:
- **Accuracy**: 99.84%
- **F1-Score**: 99.83%
- **Loss**: 0.0065

## Technical Details

### Text Preprocessing
- Convert text to lowercase
- Remove email addresses, numbers, and URLs
- Remove HTML tags
- Remove stopwords using NLTK
- Apply Porter stemming

### Model Architecture
- **Base Model**: BERT-base-uncased
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Configuration**:
  - Rank (r): 8
  - Alpha: 32
  - Dropout: 0.1
  - Task Type: Sequence Classification

### Training Configuration
- **Epochs**: 3
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Weight Decay**: 0.01
- **Max Sequence Length**: 512 tokens

## Dependencies

The project requires the following Python packages:
- `streamlit` - Web application framework
- `torch` - PyTorch for deep learning
- `transformers` - Hugging Face transformers library
- `pandas` - Data manipulation
- `nltk` - Natural language processing
- `scikit-learn` - Machine learning utilities
- `peft` - Parameter Efficient Fine-Tuning (LoRA)

## API Usage

The prediction function can be used programmatically:

```python
from app import predict_news

# Example usage
news_text = "Your news article text here..."
result = predict_news(news_text)

if result == 1:
    print("The news is real")
elif result == 0:
    print("The news is fake")
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library and BERT model
- [Streamlit](https://streamlit.io/) for the web application framework
- The dataset providers for the fake news detection dataset

## Contact

For questions or suggestions, please open an issue in the repository.
