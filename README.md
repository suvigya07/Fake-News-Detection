# Fake News / Misinformation Detection System
MCA Project — Graphic Era University, 2026

## Project Structure

```
fakenews/
├── data/
│   └── WELFake_Dataset.csv        ← download this (see below)
├── ml/
│   ├── utils.py                   ← shared text cleaning
│   ├── train.py                   ← train and save model
│   ├── predict.py                 ← called by server, returns JSON
│   └── model/                     ← auto-created after training
│       ├── model.pkl
│       └── vectorizer.pkl
├── backend/
│   ├── server.js                  ← Express API
│   └── package.json
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── README.md
```

---

## Setup (Windows)

### 1. Download the dataset

Go to: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

Download `WELFake_Dataset.csv` and place it in the `data/` folder.

### 2. Install Python dependencies

```bash
pip install pandas scikit-learn nltk joblib
```

### 3. Train the model (run once)

```bash
cd ml
python train.py
```

This will print accuracy and save model files to `ml/model/`.
Expected accuracy: ~95–97%

### 4. Install Node dependencies

```bash
cd backend
npm install
```

### 5. Start the server

```bash
cd backend
npm start
```

### 6. Open in browser

Visit: http://localhost:3000

---

## How it works

1. User pastes a news article into the textarea
2. Browser sends POST request to Express at `/api/predict`
3. Express spawns Python `predict.py` via stdin/stdout
4. Python cleans the text, runs it through the trained TF-IDF + Logistic Regression model
5. Returns JSON: label (FAKE/REAL), confidence score, and top influential words
6. Frontend renders the verdict, animated gauge, word highlights, and annotated text

## Notes

- Model never checks facts — it identifies linguistic patterns of misinformation
- ~95%+ accuracy on WELFake test split
- LIME-style word importance: green words push toward "real", red toward "fake"
