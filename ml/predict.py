import sys
import json
import os
import joblib
import re
import numpy as np

from utils import clean_text

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

def load_model():
    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    vec_path   = os.path.join(MODEL_DIR, 'vectorizer.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run train.py first.")
    model      = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

def get_top_words(text: str, vectorizer, model, n: int = 10):
    """Return top n words that pushed prediction toward fake or real."""
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]  # positive = real, negative = fake

    # Only look at words present in this document
    nonzero_idx = vec.nonzero()[1]
    word_scores = []
    for idx in nonzero_idx:
        word_scores.append({
            'word': feature_names[idx],
            'score': float(coefs[idx]),
            'tfidf': float(vec[0, idx])
        })

    word_scores.sort(key=lambda x: abs(x['score'] * x['tfidf']), reverse=True)
    top = word_scores[:n]

    highlights = []
    for item in top:
        direction = 'real' if item['score'] > 0 else 'fake'
        highlights.append({
            'word': item['word'],
            'direction': direction,
            'strength': round(abs(item['score'] * item['tfidf']), 4)
        })
    return highlights

def predict(text: str):
    model, vectorizer = load_model()
    cleaned = clean_text(text)

    if not cleaned.strip():
        return {
            'error': 'Text too short or contains no meaningful content after cleaning.'
        }

    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]  # [P(fake), P(real)]
    label_idx = int(np.argmax(proba))
    # label = 'REAL' if label_idx == 1 else 'FAKE'
    label = 'FAKE' if label_idx == 1 else 'REAL'
    confidence = round(float(proba[label_idx]) * 100, 1)

    highlights = get_top_words(text, vectorizer, model, n=12)

    return {
        'label': label,
        'confidence': confidence,
        'fake_probability': round(float(proba[0]) * 100, 1),
        'real_probability': round(float(proba[1]) * 100, 1),
        'highlights': highlights,
        'word_count': len(text.split())
    }

if __name__ == '__main__':
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
        text = payload.get('text', '')
        result = predict(text)
    except Exception as e:
        result = {'error': str(e)}
    print(json.dumps(result))
