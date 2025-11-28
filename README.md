# Movie Recommender System using SVD (MovieLens-32M)

This repository contains the complete implementation of a matrix-factorization–based movie recommender system built using **Singular Value Decomposition (SVD)** on the **MovieLens-32M** dataset.
The project covers data preprocessing, feature engineering, exploratory analysis, model development, evaluation, and recommendation generation.
All methods and results correspond to the project report included in this repository.

---

## 1. Project Overview

The goal of this project is to develop a scalable collaborative filtering model capable of predicting user preferences and generating relevant movie recommendations.
The system processes over **32 million** user–movie interactions and builds an SVD model using the **Surprise** library.


* Total ratings after cleaning: **32,000,204**
* Engineered features: **20 genre indicators** + **rating_year**
* Model: Surprise SVD
* Evaluation protocol: **80/20 train–test split**

**Final model performance:**

| Metric       | Score  |
| ------------ | ------ |
| RMSE         | 0.8259 |
| MAE          | 0.6243 |
| Precision@10 | 0.6625 |
| Recall@10    | 0.6679 |

---

## 2. Data Pipeline

### 2.1 Preprocessing



* Merging of `ratings.csv` and `movies.csv`
* Removal of ~3,153 null entries
* One-hot encoding of 20 genres
* Conversion of Unix timestamps to datetime
* Extraction of `rating_year` feature
* Final dataset saved as `cleaned_movies.csv`

### 2.2 Exploratory Data Analysis


* Rating distribution analysis
* Yearly activity trends (1995–2023)
* Most-reviewed films
* Genre frequencies and co-occurrence correlations

---

## 3. Model Development

### 3.1 Training

The final SVD model uses:

* `n_factors = 50`
* `n_epochs = 10`
* `lr_all = 0.005`
* `reg_all = 0.04`
* Train/test split: 80/20

### 3.2 Evaluation

* RMSE and MAE for rating prediction
* Precision@10 and Recall@10 for Top-N relevance evaluation
* Qualitative examples of recommendations
* Cold-start fallback to popularity-based suggestions

Hyperparameter tuning experiments include GridSearchCV and RandomizedSearch on 5–10% samples.

---

## 4. Repository Structure

```
├── data/
│   ├── raw/                     # Original MovieLens files
│   ├── cleaned_movies.csv       # Preprocessed dataset
│
├── notebooks/
│   ├── Preprocessing and EDA.ipynb
│   ├── Model development & Performance Analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── svd_model.py
│   ├── metrics.py
│   └── recommend.py
│
├── report/
│   └── DMS672_Group8_Report.pdf
│
└── README.md
```

---

## 5. Usage

### 5.1 Install Dependencies

```
pip install -r requirements.txt
```

### 5.2 Preprocess Data

```
python src/preprocessing.py
```

### 5.3 Train SVD Model

```
python src/svd_model.py
```

### 5.4 Generate Recommendations

```
python src/recommend.py
```

---


If you'd like, I can also provide a **requirements.txt**, a **.gitignore**, or convert this into a **LaTeX README** for academic submissions.
