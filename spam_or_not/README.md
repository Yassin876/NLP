# Spam or Not Spam Classifier

A project for classifying email messages as Spam or Not Spam using Python, Streamlit, and scikit-learn.

## Requirements

- Python 3.8+
- All packages listed in requirements.txt

Install requirements:
```bash
pip install -r requirements.txt
```

## How to Run

1. Make sure the following files are in the same directory:
   - `app.py`
   - `the model.pkl` (the trained model)
   - `Vectorizer.pkl` (the TfidfVectorizer object)

2. Run the app:
```bash
streamlit run app.py
```

3. Enter the email text in the box and click "predict".

## Retrain the Model (Optional)

If you want to retrain the model or update the vectorizer:

1. Open the notebook `spam_or_notspam.ipynb`.
2. Run all cells to the end.
3. Make sure the last cell contains:
```python
joblib.dump(model, 'the model.pkl')
joblib.dump(tfidf, 'Vectorizer.pkl')
```
4. Copy the generated files (`the model.pkl`, `Vectorizer.pkl`) to your project directory.

## Notes
- If you get an error like: `'csr_matrix' object has no attribute 'transform'`, it means your `Vectorizer.pkl` file does not contain the correct vectorizer object. Save it again as shown above.
- You can modify the user interface in `app.py` as you wish.

## Contact
For any questions or further development, contact the developer or open an Issue in the repo.
