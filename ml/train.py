import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import glob

from utils import clean_text

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
LOCAL_DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'WELFake_Dataset.csv')

def get_dataset_path():
    # 1. Use local file if already present
    if os.path.exists(LOCAL_DATA):
        print(f"  Found local dataset at {LOCAL_DATA}")
        return LOCAL_DATA

    # 2. Download via kagglehub
    print("  Local dataset not found. Downloading from Kaggle...")
    print("  (requires kaggle API token — see https://www.kaggle.com/settings > API)")
    try:
        import kagglehub
        path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
        print(f"  Downloaded to: {path}")
        # kagglehub puts files inside the returned path directory
        matches = glob.glob(os.path.join(path, "**", "WELFake_Dataset.csv"), recursive=True)
        if not matches:
            raise FileNotFoundError(f"WELFake_Dataset.csv not found inside {path}")
        return matches[0]
    except ImportError:
        raise ImportError(
            "kagglehub not installed. Run:  pip install kagglehub\n"
            "Or manually place WELFake_Dataset.csv in the data/ folder."
        )

def load_data():
    print("Loading dataset...")
    csv_path = get_dataset_path()
    df = pd.read_csv(csv_path)
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
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"{'='*50}\n")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,      os.path.join(MODEL_DIR, 'model.pkl'))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'vectorizer.pkl'))
    print(f"Model saved to {MODEL_DIR}/")

if __name__ == '__main__':
    train()
