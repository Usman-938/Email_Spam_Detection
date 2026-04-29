# 📧 Email Spam Detection with Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A complete machine learning project that detects **email spam** using Natural Language Processing (NLP) and three classification algorithms. The best model — **Multinomial Naïve Bayes** — achieves **98.21% accuracy**.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Methodology](#-methodology)
- [Results](#-results)
- [Model Saving & Loading](#-model-saving--loading)
- [Sample Predictions](#-sample-predictions)
- [Author](#-author)

---

## 🧠 Project Overview

Email spam is a major cybersecurity and productivity problem. This project builds an automated spam classifier using classical ML algorithms on a real-world SMS/email dataset. The pipeline covers:

1. Data loading and exploration
2. Text preprocessing with NLP (regex cleaning, tokenization, stopword removal, stemming)
3. Feature extraction using Bag of Words (CountVectorizer)
4. Training 3 classifiers: Random Forest, Decision Tree, Multinomial Naïve Bayes
5. Model evaluation with Confusion Matrix, Accuracy, and Classification Report
6. Saving all models with Pickle for future deployment

---

## 📂 Dataset

**Name:** Email/SMS Spam Collection Dataset  
**Source:** [Kaggle — Email Spam Detection](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)

| Property | Value |
|----------|-------|
| Total Records | 5,572 |
| Features Used | `label` (ham/spam), `message` (text) |
| Ham (Legitimate) | 4,825 |
| Spam | 747 |

> **Download Instructions:**
> 1. Visit the Kaggle link above
> 2. Download `spam.csv`
> 3. Place it in the same folder as the notebook

---

## 📁 Project Structure

```
Email-Spam-Detection/
│
├── Email_Spam_Detection.ipynb   # Main Jupyter Notebook (complete project)
├── spam.csv                     # Dataset (download from Kaggle)
├── RFC.pkl                      # Saved Random Forest Classifier
├── DTC.pkl                      # Saved Decision Tree Classifier
├── MNB.pkl                      # Saved Multinomial Naïve Bayes (Best Model)
├── CountVectorizer.pkl          # Saved CountVectorizer (for inference)
├── class_distribution.png      # Bar chart: Ham vs Spam count
├── class_proportion.png        # Pie chart: Spam vs Ham proportion
├── confusion_matrix_mnb.png    # Heatmap: MNB confusion matrix
├── model_comparison.png        # Bar chart: Accuracy comparison
└── README.md                   # Project documentation (this file)
```

---

## 🛠️ Technologies Used

| Category | Libraries / Tools |
|----------|------------------|
| Language | Python 3.8+ |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| NLP | NLTK (stopwords, PorterStemmer), RegEx |
| Machine Learning | Scikit-Learn |
| Model Persistence | Pickle |
| Environment | Jupyter Notebook / Google Colab |

---

## ⚙️ Installation

### 1. Clone or Download the Project

```bash
git clone https://github.com/Usman-938/email-spam-detection.git
cd email-spam-detection
```

### 2. Install Required Packages

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk
```

### 3. Download NLTK Stopwords (runs automatically in notebook)

```python
import nltk
nltk.download('stopwords')
```

---

## ▶️ How to Run

### Option A — Locally (Jupyter Notebook)

```bash
jupyter notebook Email_Spam_Detection.ipynb
```

Make sure `spam.csv` is in the same directory.

### Option B — Google Colab

1. Upload `Email_Spam_Detection.ipynb` to Colab
2. Mount Google Drive and upload `spam.csv`
3. Update the dataset path in the notebook:
   ```python
   spam = pd.read_csv("/content/drive/My Drive/spam.csv", encoding='ISO-8859-1')
   ```
4. Run all cells

---

## 🔬 Methodology

### Step-by-Step Pipeline

```
Raw Dataset (spam.csv)
        │
        ▼
  Data Exploration
  (null check, shape, class distribution)
        │
        ▼
  Feature Selection
  (keep 'label' and 'message' columns)
        │
        ▼
  NLP Preprocessing
  ┌─────────────────────────────┐
  │ 1. Remove non-alphabetical  │
  │ 2. Lowercase conversion     │
  │ 3. Tokenization             │
  │ 4. Stopword removal (NLTK)  │
  │ 5. Porter Stemming          │
  └─────────────────────────────┘
        │
        ▼
  Bag of Words (CountVectorizer)
  max_features = 4000
        │
        ▼
  Train-Test Split (80/20)
        │
        ▼
  Train 3 Models
  ┌──────────────────────────┐
  │ RFC — Random Forest      │
  │ DTC — Decision Tree      │
  │ MNB — Multinomial NB     │
  └──────────────────────────┘
        │
        ▼
  Evaluate (Confusion Matrix,
  Accuracy, Classification Report)
        │
        ▼
  Save Best Model (MNB.pkl)
```

---

## 📊 Results

### Model Accuracy

| Model | Accuracy |
|-------|----------|
| Random Forest Classifier | 97.31% |
| Decision Tree Classifier | 97.49% |
| **Multinomial Naïve Bayes** | **98.21% 🏆** |

### Classification Report — Best Model (MNB)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Ham   | 0.99      | 0.99   | 0.99     | 965     |
| Spam  | 0.93      | 0.93   | 0.93     | 150     |
| **Accuracy** | | | **0.98** | **1115** |

### Confusion Matrix (MNB)

```
              Predicted
              Ham   Spam
Actual  Ham  [ 955    10 ]
        Spam [  10   140 ]
```

> **Multinomial Naïve Bayes wins** because it achieves the highest accuracy and the best balanced F1-score for spam detection — correctly identifying 140 out of 150 spam emails.

---

## 💾 Model Saving & Loading

All models are saved with Pickle after training.

### Loading a saved model

```python
import pickle

# Load the best model (MNB)
with open('MNB.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the vectorizer
with open('CountVectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)
```

---

## 🧪 Sample Predictions

```python
def predict_spam(message):
    # Preprocess + vectorize + predict
    ...
    return 'SPAM 🚫' or 'HAM ✅'

predict_spam("Congratulations! You won a FREE iPhone. Click to claim!")
# → SPAM 🚫

predict_spam("Hey, are you coming to the meeting tomorrow?")
# → HAM ✅
```

---

## 👤 Author

**Muhammad Usman Ilyas**  
BCS Artificial Intelligence — Semester _(Your Semester)_  
Abdul Wali Khan University Mardan (AWKUM), Pakistan  

- 🐙 GitHub: [@Usman-938](https://github.com/Usman-938)  
- 💼 LinkedIn: [m-usman-i938](https://linkedin.com/in/m-usman-i938)  

---

## 📄 License

This project is for **educational purposes** as part of the Programming for AI course at AWKUM.  
Feel free to use and modify with proper attribution.

---

*⭐ If you found this helpful, consider starring the repository!*
