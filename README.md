# TruthMiner: Fake News Detection with NLP & Transformers

> ## 👀 What is TruthMiner?
> In today’s world, **misinformation travels faster than facts**.  
> TruthMiner is an AI-powered system that **detects fake news headlines** using both **classic Machine Learning** and **modern Transformer models (BERT)**.  
> Think of it as your **digital truth detector** — analyzing headlines and predicting whether they’re **real** or **fake**.

---

## 📌 Project Overview
**TruthMiner** is a Natural Language Processing (NLP) project designed to automatically classify news headlines as either **real** or **fake**.  
The project combines **classical machine learning models** with a **fine-tuned Transformer (DistilBERT)** to benchmark performance and highlight the trade-offs between traditional approaches and modern deep learning.

---

## 🎯 Goals
- Preprocess raw text into clean, usable features
- Apply feature engineering with **TF-IDF** and embeddings
- Train and compare multiple ML classifiers:
  - Logistic Regression
  - Naïve Bayes
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
- Fine-tune **DistilBERT** for sequence classification
- Evaluate models using **Accuracy, Precision, Recall, F1-score**
- Generate predictions for the test dataset in the original file format

---

## 📂 Dataset
- **Training data:** `dataset/training_data.csv`
  - Format: `label<TAB>headline`
  - `label = 0` → Fake News
  - `label = 1` → Real News
- **Testing data:** `dataset/testing_data.csv`
  - Same format, but labels are placeholders (`2`)
  - Your model predicts and replaces them with `0` or `1`

---

## ⚙️ Workflow

### 1️⃣ Preprocessing
- Convert text to lowercase
- Remove punctuation
- Tokenize text
- Remove stopwords
- Apply stemming / lemmatization

### 2️⃣ Feature Engineering
- TF-IDF vectorization (unigrams + bigrams)
- Embeddings for Transformer input

### 3️⃣ Modeling
- Train classical ML models with TF-IDF features
- Fine-tune **DistilBERT** using Hugging Face `Trainer` API
- Apply stratified splits (80/20) for validation

### 4️⃣ Evaluation
Metrics reported:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

### 5️⃣ Prediction
- Replace `2` labels in the test set with predicted values
- Save results as `testing_predictions.csv` (tab-separated, no header)

---

## 📊 Results (Validation Set)
| Model                  | Accuracy | Precision | Recall | F1   |
|------------------------|----------|-----------|--------|------|
| Logistic Regression     | 0.94     | 0.94      | 0.94   | 0.94 |
| Naïve Bayes             | 0.91     | 0.90      | 0.91   | 0.90 |
| Random Forest           | 0.89     | 0.88      | 0.89   | 0.88 |
| SVM (Linear)            | **0.95** | 0.94      | 0.95   | 0.95 |
| XGBoost                 | 0.93     | 0.93      | 0.93   | 0.93 |
| DistilBERT (fine-tuned) | 0.96     | 0.96      | 0.96   | 0.96 |

> ⚡ Transformers slightly outperformed classical ML, but SVM and Logistic Regression remain strong baselines.

---

## 🛠️ Tech Stack
- **Languages:** Python 3.11
- **Libraries:**
  - `pandas`, `numpy`, `scikit-learn`
  - `xgboost`
  - `torch`, `transformers`, `datasets`
- **Environment:** macOS M3 Air, VS Code
- **Hardware:** CPU + Apple MPS (GPU acceleration)

---

## 🚀 Usage
### Clone the repo
```bash
  git clone https://github.com/yourusername/truthminer.git
  cd truthminer
```
Create and activate a virtual environment (recommended):
```
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```
Install dependencies:
```
pip install -r requirements.txt
```
Repo Structure:
## Project Structure
```bash
truthminer/
├── assets/                        # Images, banners, and figures for README
│   └── truthminer_banner.png
│
├── dataset/                       # Data files
│   ├── training_data.csv          # Provided training set (label, headline)
│   ├── testing_data.csv           # Provided test set (labels=2, to be predicted)
│   └── testing_predictions.csv    # Model predictions output
│
├── project_truth_miner.ipynb      # Main Jupyter notebook (EDA, ML, DistilBERT fine-tuning)
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── LICENSE                        # License file
