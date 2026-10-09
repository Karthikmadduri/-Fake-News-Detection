# Fake News Detection using Machine Learning

A complete Machine Learning and NLP web application that classifies news articles as **Real** or **Fake**.

---

## 🚀 How to Run on Localhost

### 1. Requirements
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
Open your terminal in the project directory and run:
```bash
python -m pip install -r requirements.txt
```

### 3. Run the Web Application
Start the Flask application:
```bash
python app.py
```

### 4. Open in Browser
Open your browser and navigate to:
- **[http://localhost:5000](http://localhost:5000)** (or [http://127.0.0.1:5000](http://127.0.0.1:5000))

Paste any news article headline or text, or click one of the quick test sample buttons, then click **Analyze Article** to see the prediction and confidence score!

---

## 📁 Project Structure

| File / Folder | Description |
| :--- | :--- |
| `app.py` | Flask web application server and prediction API |
| `best_model.joblib` | Pre-trained model pipeline (CountVectorizer + Logistic Regression) |
| `templates/` | HTML web pages (`home.html`, `result.html`) |
| `train_model.py` | Script to train and compare algorithms (Logistic Regression, Naive Bayes, Linear SVM) |
| `requirements.txt` | Python package dependencies |
| `FakeNewsDetection.ipynb` | Jupyter notebook for exploratory data analysis and model experimentation |

---

## 🧠 Model Training (Optional)
A pre-trained model (`best_model.joblib`) is already included, so retraining is **not required** to run the app.

If you want to retrain the models:
1. Download the Kaggle dataset: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Place `Fake.csv` and `True.csv` into a folder named `Data/`
3. Run:
   ```bash
   python train_model.py
   ```
