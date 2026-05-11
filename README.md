# 🧠 Autism Spectrum Disorder (ASD) Screening Prediction System

A Machine Learning and Artificial Neural Network (ANN) based system to predict the likelihood of Autism Spectrum Disorder in children, built as part of a Master degree thesis of Mrs. Chhayachabbi Jha for the degree, MScISE at PUSOE under Purbanchal University

---

## 👤 Authors
**Chhayachabbi Jha**
** Prof. Raj Kumar Thakur** ( Thesis Supervisor ) 
Email: rajkshiva59@gmail.com  
GitHub: [rajaramayan](https://github.com/rajaramayan)

---

## 📌 Project Overview

This project trains and evaluates multiple ML models and an ANN on ASD screening data. The best-performing model can then be used for real-time predictions via an interactive Streamlit web application.

---

## 🗂️ Project Structure

```
├── ASD in Children using Machine Learning and ANN (1).py   # Local training script
├── streamlit_app.py                                         # Streamlit web app
├── Autism_Screening_Data_Combined.csv                       # Dataset (6075 records)
├── models/                                                  # Saved model artifacts
│   ├── trained_models.pkl                                   # All 8 ML models
│   ├── ann.pkl                                              # ANN (MLPClassifier)
│   ├── scaler.pkl                                           # StandardScaler
│   ├── le_dict.pkl                                          # Label Encoders
│   ├── results_df.pkl                                       # Metrics results
│   ├── roc_data.pkl                                         # ROC curve data
│   └── metadata.pkl                                         # Feature metadata
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## 🤖 Models Used

| Model | Type |
|---|---|
| Logistic Regression | Classical ML |
| Decision Tree | Classical ML |
| Random Forest | Classical ML |
| K-Nearest Neighbors (KNN) | Classical ML |
| SVM (Polynomial Kernel) | Classical ML |
| SVM (RBF Kernel) | Classical ML |
| Naive Bayes | Classical ML |
| QDA | Classical ML |
| ANN (MLPClassifier) | Neural Network |

---

## 📊 Model Results (Test Set)

| Model | Accuracy | F1 Score | ROC-AUC | Overfitting |
|---|---|---|---|---|
| **ANN** | **95.41%** | **0.9262** | **0.9962** | Mild |
| Random Forest | 93.31% | 0.8974 | 0.9924 | None |
| Logistic Regression | 90.92% | 0.8652 | 0.9749 | None |
| SVM (RBF) | 89.68% | 0.8500 | 0.9805 | Mild |
| Naive Bayes | 89.01% | 0.8409 | 0.9638 | None |
| Decision Tree | 87.48% | 0.8065 | 0.9358 | None |
| QDA | 86.62% | 0.8153 | 0.9716 | Mild |
| KNN | 83.56% | 0.7823 | 0.9800 | Moderate |
| SVM (Poly) | 77.34% | 0.6934 | 0.8910 | Mild |

> **Best Model: ANN** — ROC-AUC: 0.9962, Accuracy: 95.41%

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models Locally
```bash
python "ASD in Children using Machine Learning and ANN (1).py"
```
This trains all 9 models, prints evaluation metrics, shows plots, and saves all model artifacts to the `models/` folder.

### 3. Launch the Streamlit App
```bash
streamlit run streamlit_app.py
```
Go to **🤖 Model Training** → click **📂 Load Pre-trained Models**.

---

## 🌐 Streamlit App Features

- **🏠 Home** — Project overview and usage guide
- **🤖 Model Training** — Load pre-trained models or train in-browser
- **🔮 Make Prediction** — Enter patient screening values and get ASD prediction
- **📊 Model Comparison** — ROC curves, bar charts, performance metrics table

---

## 🧪 Dataset

- **File:** `Autism_Screening_Data_Combined.csv`
- **Records:** 6,075
- **Features:** 14 (A1–A10 screening questions, Age, Sex, Jaundice, Family ASD history)
- **Target:** `Class` (ASD Positive / Negative)
- **Imbalance handling:** SMOTE (Synthetic Minority Over-sampling Technique)

---

## 🛠️ Tech Stack

- Python 3.10
- scikit-learn, imbalanced-learn
- Streamlit
- NumPy, Pandas, Matplotlib, Seaborn

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. It is not a substitute for professional medical diagnosis.
