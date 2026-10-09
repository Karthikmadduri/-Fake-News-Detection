"""
Train and compare three text-classification models for Fake News Detection.
Required files (optional for retraining):
    Data/Fake.csv
    Data/True.csv

Dataset source:
    Kaggle Fake and Real News Dataset:
    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Note:
    A pre-trained model (best_model.joblib) is already included with this project!
    To run the web demo immediately, run:
        python app.py
"""

import re
import string
import nltk
import joblib
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

def load_stop_words():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))

STOP_WORDS = load_stop_words()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)      # remove digits
    return " ".join(word for word in text.split() if word not in STOP_WORDS)

def main():
    fake_path = DATA_DIR / "Fake.csv"
    true_path = DATA_DIR / "True.csv"
    
    if not fake_path.exists() or not true_path.exists():
        print("\n" + "=" * 60)
        print(" [INFO] Training dataset files not found!")
        print(" Expected locations:")
        print(f"   - {fake_path}")
        print(f"   - {true_path}")
        print("\n To retrain the models:")
        print("   1. Download the dataset from Kaggle:")
        print("      https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print("   2. Place Fake.csv and True.csv inside the 'Data' folder.")
        print("   3. Run 'python train_model.py' again.")
        print("\n NOTE: You DO NOT need to retrain to run the web application!")
        print(" The trained model 'best_model.joblib' is already ready.")
        print(" Simply run:")
        print("   python app.py")
        print("=" * 60 + "\n")
        return

    print("Loading datasets from Data folder...")
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    if "text" not in fake_df.columns or "text" not in true_df.columns:
        raise ValueError("Both CSV files must contain a 'text' column.")

    fake_df["target"] = "fake"
    true_df["target"] = "true"
    
    data = pd.concat([fake_df[["text", "target"]], true_df[["text", "target"]]], ignore_index=True)
    data = data.dropna(subset=["text"])
    
    print(f"Preprocessing {len(data):,} articles (lowercasing, removing punctuation & stopwords)...")
    data["clean_text"] = data["text"].apply(preprocess_text)
    data = data[data["clean_text"].str.strip().ne("")]

    print("Splitting train and test sets (60% train / 40% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        data["clean_text"], data["target"], test_size=0.4, random_state=42, stratify=data["target"]
    )
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": SVC(kernel="linear", probability=True),
    }

    best_name, best_accuracy, best_pipeline = None, -1.0, None
    print(f"\nTraining set size: {len(X_train):,} | Testing set size: {len(X_test):,}")
    print("\nEvaluating models:")
    print("-" * 75)
    
    for name, classifier in models.items():
        pipeline = Pipeline([("vect", CountVectorizer()), ("model", classifier)])
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, predicted)
        precision = precision_score(y_test, predicted, pos_label="true", zero_division=0)
        recall = recall_score(y_test, predicted, pos_label="true", zero_division=0)
        f1 = f1_score(y_test, predicted, pos_label="true", zero_division=0)
        
        print(f" {name:24s} | Acc: {accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")
        if accuracy > best_accuracy:
            best_name, best_accuracy, best_pipeline = name, accuracy, pipeline

    model_dest = BASE_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, model_dest)
    print("-" * 75)
    print(f"\n[SUCCESS] Best model '{best_name}' (Accuracy: {best_accuracy:.4f}) saved to:\n  {model_dest}")
    print("\nYou can now launch the web app with:")
    print("  python app.py\n")

if __name__ == "__main__":
    main()