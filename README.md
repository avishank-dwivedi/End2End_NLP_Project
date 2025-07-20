# 😄 End-to-End NLP Project – Emotion Detection from Text

This project detects human **emotions** (like joy, anger, fear, sadness, etc.) from raw text using Natural Language Processing (NLP) and Machine Learning/Deep Learning techniques.

---

## 🚀 Features

- Text preprocessing with NLTK
- Multi-class emotion classification
- Trained using Logistic Regression / LSTM / Random Forest (based on config)
- TF-IDF / Word Embedding based input features
- Streamlit interface for real-time emotion prediction
- Model evaluation & performance metrics included

---

## 🧠 Technologies Used

- Python
- Pandas & NumPy
- NLTK / spaCy
- Scikit-learn / TensorFlow / Keras
- Streamlit
- Matplotlib / Seaborn

---



| Text | Emotion |
|------|---------|
| I feel happy today | joy |
| I’m scared | fear |
| This is frustrating | anger |

> Emotions include: `joy`, `anger`, `fear`, `sadness`, `love`, `surprise`, etc.

---

## 🧹 Text Preprocessing Steps

- Lowercasing  
- Removing punctuation/special characters  
- Stopword removal  
- Lemmatization / Stemming  
- Tokenization  
- Feature extraction (TF-IDF or word embeddings)

---

## 🔍 Model Training

You can choose:
- `LogisticRegression` for light training
- `RandomForestClassifier` for robustness
- `LSTM / Bi-LSTM` if using word embeddings (deep learning)

Example training with logistic regression:

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
0
