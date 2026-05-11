# Thesis Report

## Early Detection of Autism Spectrum Disorder in Children Using Machine Learning and Artificial Neural Networks

---

**Author:** Raj Kumar Thakur  
**Email:** rajkshiva59@gmail.com  
**Date:** May 2026

---

## Abstract

Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition characterised by persistent challenges in social communication, restricted and repetitive patterns of behaviour, and sensory processing differences. Early and accurate diagnosis of ASD is critical, as timely intervention significantly improves developmental outcomes for affected children. However, traditional diagnostic procedures are time-consuming, resource-intensive, and often subject to long waiting periods, limiting access especially in underserved regions.

This thesis proposes a machine learning and artificial neural network (ANN) based computational framework for the early screening and prediction of ASD in children. The study utilises a combined ASD screening dataset comprising 6,075 records with 14 clinically relevant features, including ten behavioural screening questions (A1–A10), age, sex, history of jaundice, and family history of ASD. The target variable is a binary classification label indicating ASD positive or ASD negative.

A rigorous data preprocessing pipeline was implemented, including duplicate removal, missing value imputation using mode strategy, categorical variable encoding using individual Label Encoders, and class imbalance correction using the Synthetic Minority Over-sampling Technique (SMOTE). Feature scaling was performed using StandardScaler to normalise the input space prior to model training.

Nine classification models were trained and evaluated: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbours (KNN), Support Vector Machine with Polynomial kernel (SVM-Poly), Support Vector Machine with RBF kernel (SVM-RBF), Naïve Bayes, Quadratic Discriminant Analysis (QDA), and a Multilayer Perceptron Artificial Neural Network (MLP-ANN). The MLP-ANN was constructed with two hidden layers of 32 and 16 neurons respectively, using ReLU activation, trained via the Adam optimiser with backpropagation over 200 iterations using Binary Cross-Entropy loss.

Model performance was assessed using Accuracy, Precision, Recall, Specificity, F1 Score, ROC-AUC, and Log Loss on a held-out test set (20% split). Overfitting analysis was conducted by comparing training and test accuracy gaps for all models.

Experimental results demonstrate that the MLP-ANN achieved the highest performance with a ROC-AUC of **0.9962**, test accuracy of **95.41%**, F1 Score of **0.9262**, and a mild overfitting gap of 3.18%, indicating strong generalisation capability. Random Forest ranked second with a ROC-AUC of 0.9924 and zero overfitting, followed by Logistic Regression with ROC-AUC of 0.9749.

The trained models were serialised using Python's pickle library and deployed as an interactive web application using Streamlit, enabling real-time ASD risk prediction based on user-provided screening inputs. The application provides predictions from both the best classical model and the ANN, along with comprehensive model comparison visualisations including ROC curves, performance bar charts, and overfitting analysis tables.

This work demonstrates that machine learning and ANN-based approaches can serve as reliable, scalable, and accessible screening tools for early ASD detection, potentially aiding clinicians and caregivers in identifying at-risk children at a much earlier stage than traditional methods allow.

**Keywords:** Autism Spectrum Disorder, ASD Screening, Machine Learning, Artificial Neural Network, MLP Classifier, Backpropagation, SMOTE, Classification, Early Detection, Streamlit Deployment.

---
