import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import sys

from utils import clean_text

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'WELFake_Dataset.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

def load_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Raw shape: {df.shape}")

    # WELFake columns: Unnamed:0, title, text, label  (0=fake, 1=real)
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)

    # Combine title + text for richer features
    df['title'] = df['title'].fillna('')
    df['content'] = df['title'] + ' ' + df['text']
    df['content'] = df['content'].apply(clean_text)

    df = df[df['content'].str.strip() != '']
    print(f"  After cleaning: {df.shape}")
    print(f"  Label distribution:\n{df['label'].value_counts()}")
    return df

def train():
    df = load_data()

    X = df['content']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

    print("\nFitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    print("Training Logistic Regression...")
    model = LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver='lbfgs',
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Real','Fake'])}")
    # print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Fake','Real'])}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"{'='*50}\n")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,      os.path.join(MODEL_DIR, 'model.pkl'))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'vectorizer.pkl'))
    print(f"Model saved to {MODEL_DIR}/")

if __name__ == '__main__':
    train()
