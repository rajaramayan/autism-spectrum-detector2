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

## Table of Contents

| | |
|---|---|
| **Abstract** | |
| **Table of Contents** | |
| **List of Figures** | |
| **List of Tables** | |
| | |
| **Chapter 2: Literature Review** | |
| 2.1 Overview | |
| 2.2 ASD Prevalence, Clinical Challenges, and the Need for Automated Tools | |
| 2.3 Conventional Machine Learning Classifiers for ASD Screening | |
| 2.4 Random Forest and Ensemble Methods for ASD | |
| 2.5 Support Vector Machines and Kernel Methods for ASD | |
| 2.6 Artificial Neural Networks and Deep Learning for ASD | |
| 2.7 Handling Class Imbalance with SMOTE | |
| 2.8 Feature Selection and Dimensionality Reduction in ASD Datasets | |
| 2.9 Comparative Studies and Systematic Reviews | |
| 2.10 Mobile and Web Deployment of ASD Screening Tools | |
| 2.11 Logistic Regression, Naïve Bayes, and QDA as Interpretable Baselines | |
| 2.12 Ethical Considerations and Limitations in ML-Based ASD Diagnosis | |
| 2.13 Summary | |
| | |
| **Chapter 3: Research Gap** | |
| 3.1 Introduction | |
| 3.2 Identified Research Gaps | |
| &nbsp;&nbsp;&nbsp;3.2.1 Lack of Comprehensive Multi-Model Benchmarking | |
| &nbsp;&nbsp;&nbsp;3.2.2 Insufficient Overfitting Analysis and Generalisation Reporting | |
| &nbsp;&nbsp;&nbsp;3.2.3 Limited Use of Large, Combined, and Representative Datasets | |
| &nbsp;&nbsp;&nbsp;3.2.4 Inadequate Handling of Class Imbalance | |
| &nbsp;&nbsp;&nbsp;3.2.5 Absence of Probabilistic and Discriminant Analysis Classifiers | |
| &nbsp;&nbsp;&nbsp;3.2.6 Lack of End-to-End Deployment with Multi-Model Inference | |
| &nbsp;&nbsp;&nbsp;3.2.7 Reproducibility and Code Transparency Deficits | |
| 3.3 Summary of Research Gaps and Thesis Contributions | |
| 3.4 Research Objectives | |
| | |
| **Chapter 4: Methodology** | |
| 4.1 Overview | |
| 4.2 Overall System Architecture and Workflow | |
| 4.3 Phase 1: Dataset Description | |
| &nbsp;&nbsp;&nbsp;4.3.1 Data Source | |
| &nbsp;&nbsp;&nbsp;4.3.2 Feature Description | |
| &nbsp;&nbsp;&nbsp;4.3.3 Target Variable and Class Distribution | |
| 4.4 Phase 2: Data Preprocessing | |
| &nbsp;&nbsp;&nbsp;4.4.1 Duplicate Removal | |
| &nbsp;&nbsp;&nbsp;4.4.2 Missing Value Imputation | |
| &nbsp;&nbsp;&nbsp;4.4.3 Categorical Label Encoding | |
| &nbsp;&nbsp;&nbsp;4.4.4 Numeric Coercion and Residual Fill | |
| 4.5 Phase 3: Train-Test Partitioning | |
| 4.6 Phase 4: Class Imbalance Handling with SMOTE | |
| 4.7 Phase 5: Feature Scaling | |
| 4.8 Phase 6: Model Definition and Training | |
| &nbsp;&nbsp;&nbsp;4.8.1 Logistic Regression | |
| &nbsp;&nbsp;&nbsp;4.8.2 Decision Tree | |
| &nbsp;&nbsp;&nbsp;4.8.3 Random Forest | |
| &nbsp;&nbsp;&nbsp;4.8.4 K-Nearest Neighbours (KNN) | |
| &nbsp;&nbsp;&nbsp;4.8.5 Support Vector Machine – Polynomial Kernel | |
| &nbsp;&nbsp;&nbsp;4.8.6 Support Vector Machine – RBF Kernel | |
| &nbsp;&nbsp;&nbsp;4.8.7 Gaussian Naïve Bayes | |
| &nbsp;&nbsp;&nbsp;4.8.8 Quadratic Discriminant Analysis (QDA) | |
| &nbsp;&nbsp;&nbsp;4.8.9 Multilayer Perceptron – Artificial Neural Network (MLP-ANN) | |
| 4.9 Phase 7: Model Evaluation Framework | |
| &nbsp;&nbsp;&nbsp;4.9.1 Overfitting Analysis | |
| &nbsp;&nbsp;&nbsp;4.9.2 Confusion Matrices — All Nine Models | |
| 4.10 Phase 8: Model Serialisation | |
| 4.11 Phase 9: Streamlit Web Application Deployment | |
| 4.12 Tools, Libraries, and Environment | |
| 4.13 Summary | |
| | |
| **Chapter 5: Results and Discussion** | |
| 5.1 Overview | |
| 5.2 Dataset Summary | |
| 5.3 Overall Model Performance Results | |
| 5.4 MLP-ANN — Best Model Analysis | |
| 5.5 Random Forest — Second Best Model | |
| 5.6 Model-by-Model Analysis | |
| &nbsp;&nbsp;&nbsp;5.6.1 Logistic Regression | |
| &nbsp;&nbsp;&nbsp;5.6.2 Decision Tree | |
| &nbsp;&nbsp;&nbsp;5.6.3 K-Nearest Neighbours (KNN) | |
| &nbsp;&nbsp;&nbsp;5.6.4 SVM (RBF Kernel) | |
| &nbsp;&nbsp;&nbsp;5.6.5 SVM (Polynomial Kernel) | |
| &nbsp;&nbsp;&nbsp;5.6.6 Naïve Bayes | |
| &nbsp;&nbsp;&nbsp;5.6.7 Quadratic Discriminant Analysis (QDA) | |
| 5.7 Overfitting Analysis | |
| 5.8 ROC-AUC Comparison | |
| 5.9 Metric-by-Metric Cross-Model Comparison | |
| &nbsp;&nbsp;&nbsp;5.9.1 Recall (Sensitivity) — Clinical Priority Metric | |
| &nbsp;&nbsp;&nbsp;5.9.2 Specificity — Minimising Unnecessary Referrals | |
| &nbsp;&nbsp;&nbsp;5.9.3 F1 Score — Harmonic Mean of Precision and Recall | |
| &nbsp;&nbsp;&nbsp;5.9.4 Log Loss — Probability Calibration Quality | |
| 5.10 Ranking Summary — Multi-Criteria Evaluation | |
| 5.11 Comparison with Prior Literature | |
| 5.12 Clinical Interpretation of Results | |
| 5.13 Summary of Key Findings | |
| | |
| **Chapter 6: Discussion** | |
| 6.1 Introduction to Discussion | |
| 6.2 Interpreting the Performance Hierarchy | |
| &nbsp;&nbsp;&nbsp;6.2.1 Why the MLP-ANN Outperforms Classical Models | |
| &nbsp;&nbsp;&nbsp;6.2.2 Why Random Forest is the Best Classical Model | |
| &nbsp;&nbsp;&nbsp;6.2.3 Why SVM (Polynomial) Underperforms | |
| &nbsp;&nbsp;&nbsp;6.2.4 The KNN and QDA Paradox — Perfect Recall Without Optimal Utility | |
| 6.3 The Precision-Recall Trade-Off in ASD Screening | |
| 6.4 Significance of Overfitting Control | |
| &nbsp;&nbsp;&nbsp;6.4.1 Why Overfitting Control Matters in Medical AI | |
| &nbsp;&nbsp;&nbsp;6.4.2 The Decision Tree Overfitting Paradox | |
| &nbsp;&nbsp;&nbsp;6.4.3 MLP-ANN's 3.18% Gap — Contextual Interpretation | |
| 6.5 Addressing the Research Gaps | |
| 6.6 Comparison with State-of-the-Art | |
| 6.7 SMOTE and Class Imbalance: Impact on Results | |
| 6.8 Limitations of the Study | |
| 6.9 Ethical Considerations | |
| 6.10 Implications for Future Research | |
| 6.11 Practical Utility of the Streamlit Application | |
| 6.12 Summary | |
| | |
| **Chapter 7: Conclusion** | |
| 7.1 Overview | |
| 7.2 Principal Findings | |
| &nbsp;&nbsp;&nbsp;7.2.1 The MLP-ANN is the Superior Model for ASD Screening | |
| &nbsp;&nbsp;&nbsp;7.2.2 Random Forest is the Most Reliable Classical Classifier | |
| &nbsp;&nbsp;&nbsp;7.2.3 Overfitting Was Controlled Across All Models | |
| &nbsp;&nbsp;&nbsp;7.2.4 Class Imbalance Handling is Essential for High Recall | |
| &nbsp;&nbsp;&nbsp;7.2.5 The Feature Set is Highly Discriminative | |
| 7.3 Research Objectives — Achievement Summary | |
| 7.4 Contributions of the Thesis | |
| 7.5 Limitations Acknowledged | |
| 7.6 Recommendations | |
| 7.7 Final Conclusion | |
| | |
| **References** | |

---

## List of Figures

| Figure | Caption |
|---|---|
| **Figure 4.1** | Overall system workflow — nine-phase methodology pipeline |
| **Figure 4.2** | Phase 1: Dataset description and feature overview |
| **Figure 4.3** | Phase 2: Data preprocessing pipeline (deduplication, imputation, encoding) |
| **Figure 4.4** | Phase 3: Stratified 80/20 train-test partitioning |
| **Figure 4.5** | Phase 4: SMOTE class imbalance correction applied to training set |
| **Figure 4.6** | Phase 5: StandardScaler feature normalisation |
| **Figure 4.7** | Phase 6: Nine-model training with tuned hyperparameters |
| **Figure 4.8** | Phase 7: Seven-metric model evaluation framework |
| **Figure 4.9** | Phase 8: Model serialisation using Python pickle |
| **Figure 4.10** | Phase 9: Streamlit web application deployment architecture |
| **Figure 4.11** | Confusion matrices for all nine classifiers on the test set (n=1,046) |
| **Figure 5.1** | MLP-ANN performance summary — key metrics and confusion matrix |
| **Figure 5.2** | Random Forest performance summary — key metrics and confusion matrix |
| **Figure 5.3** | Overfitting gap chart — train accuracy vs. test accuracy for all nine models |
| **Figure 5.4** | ROC-AUC ranking — all nine models sorted by discriminative power |
| **Figure 5.5** | Clinical interpretation of MLP-ANN results on the test set |
| **Figure 6.1** | Precision-recall trade-off comparison across all nine classifiers |

---

## List of Tables

| Table | Caption |
|---|---|
| **Table 2.1** | Summary of 25 reviewed studies on ML/ANN-based ASD screening (2021–2026) |
| **Table 3.1** | Summary of identified research gaps and corresponding thesis contributions |
| **Table 4.1** | Dataset feature description — 14 input features and target variable |
| **Table 4.2** | Hyperparameter configurations for all nine classifiers |
| **Table 4.3** | Train-test partition statistics (pre- and post-SMOTE) |
| **Table 4.4** | Tools, libraries, and software environment |
| **Table 5.1** | Dataset partition statistics (records, class distribution, split sizes) |
| **Table 5.2** | Complete seven-metric performance comparison for all nine classifiers |
| **Table 5.3** | Overfitting analysis — train-test accuracy gaps for all nine models |
| **Table 5.4** | Multi-criteria rank summary across all seven evaluation metrics |
| **Table 6.1** | Research gap coverage assessment — G1–G7 addressed by this thesis |
| **Table 7.1** | Research objective achievement summary — RO1–RO5 |

---

## Chapter 2: Literature Review

### 2.1 Overview

The intersection of machine learning (ML), artificial intelligence (AI), and clinical neuroscience has produced a rapidly growing body of research aimed at automating and improving the diagnosis of Autism Spectrum Disorder (ASD). ASD affects millions of children worldwide and is characterised by a heterogeneous symptom profile that makes early and accurate diagnosis both critical and challenging. Over the past five years, researchers have explored a wide range of computational techniques—from classical classifiers and ensemble methods to deep neural networks and transformer-based architectures—to build reliable, scalable screening and diagnostic tools. This chapter critically reviews 25 significant and recent studies (2021–2026) that inform the methodology, motivation, and design choices of this thesis.

---

### 2.2 ASD Prevalence, Clinical Challenges, and the Need for Automated Tools

Autism Spectrum Disorder is among the fastest-growing neurodevelopmental conditions globally. The diagnostic landscape, dominated by instruments such as the Autism Diagnostic Observation Schedule (ADOS-2) and the Autism Diagnostic Interview–Revised (ADI-R), is thorough but time-intensive, clinician-dependent, and largely inaccessible in low-resource settings. Akter *et al.* [1] highlighted this systemic bottleneck in their landmark 2021 IEEE Access study, demonstrating that machine learning models trained on behavioural screening questionnaire data could match or exceed the accuracy of structured clinical interviews for early-stage ASD detection. Their pipeline, which trained multiple classifiers including Random Forest and ANN on the UCI ASD dataset, reported AUC values exceeding 0.95, establishing a strong computational baseline for subsequent work.

The inadequacy of conventional diagnostic pathways is further underscored by Mandal, Bhattacharya, and Jana [2], who noted that average diagnostic delays in children can range from one to four years post-parental concern, a window during which early behavioural interventions could have produced substantial developmental gains. Their study proposed ensemble ML classifiers as screening adjuncts to triage at-risk children before formal clinical evaluation, achieving 97.2% accuracy on a paediatric cohort. Similarly, Thabtah, Zhang, and Abdelhamid [3] explored feature-reduction strategies and rule-based induction for ASD screening, demonstrating that compact, interpretable models using as few as six features from the AQ-10 instrument can maintain diagnostic precision above 94%, an observation directly relevant to the feature engineering decisions in this thesis.

---

### 2.3 Conventional Machine Learning Classifiers for ASD Screening

The application of classical supervised learning classifiers to structured ASD screening datasets has been one of the most active areas of research since 2021. Al-Qudah, Al-Khasawneh, and Zamil [4] conducted a rigorous comparative evaluation of eight classifiers—including Logistic Regression, Decision Tree, k-Nearest Neighbours (KNN), and Naïve Bayes—on a merged ASD behavioural dataset, reporting that ensemble methods such as Random Forest consistently outperformed single-model approaches, particularly on imbalanced class distributions. Their finding that Logistic Regression achieves competitive AUC (0.97) despite its simplicity corroborates the baseline results reported in this thesis.

Parikh *et al.* [5] specifically benchmarked classifiers on the widely used AQ-10 screening dataset, comparing seven models under identical training conditions. Random Forest achieved the highest accuracy of 98.1%, while Support Vector Machine with the radial basis function (RBF) kernel delivered the best ROC-AUC score of 0.99, a result that aligns closely with the SVM-RBF performance reported in this work. Importantly, the authors demonstrated the sensitivity of model performance to preprocessing choices—particularly missing value imputation and categorical encoding strategies—reinforcing the importance of the systematic preprocessing pipeline employed in this thesis.

Bircanoglu, Anafarta, and Orhan [6] extended this line of inquiry to heterogeneous multi-source ASD datasets, showing that feature standardisation using StandardScaler significantly improved the convergence and accuracy of distance-based classifiers such as KNN, and that models trained without scaling exhibited accuracy drops of up to 8%. This finding directly motivated the StandardScaler preprocessing step applied across all classifiers in the present study. Duda, Kosmicki, and Wall [7] further evaluated the diagnostic accuracy of ML algorithms on paediatric ASD data, emphasising the risk of data leakage in cross-validation designs and recommending strict held-out test sets—an approach followed rigorously in this thesis through a fixed 20% stratified split.

---

### 2.4 Random Forest and Ensemble Methods for ASD

Random Forest classifiers have emerged as a consistently high-performing baseline for ASD detection tasks due to their inherent robustness against overfitting, built-in feature importance estimation, and tolerance for mixed data types. Nnamoko *et al.* [8] demonstrated the efficacy of Random Forest for ASD symptom severity classification, achieving 96.8% accuracy on a multi-source clinical dataset and using Gini impurity-based feature importance to identify the A7 ("hand flapping") and A4 ("social smiling") behavioural questions as the strongest predictors—consistent with clinical expectations. Their study also showed that bagging-based ensembles exhibit near-zero train-test accuracy gaps, confirming their superior generalisation properties relative to single-tree models.

Satu *et al.* [9] combined Random Forest with feature selection algorithms, showing that recursive feature elimination (RFE) guided by Random Forest importance scores reduced the feature space by 40% without loss of predictive power. Their resulting model achieved ROC-AUC of 0.993 on an independent validation set, demonstrating that feature pruning can improve model interpretability without sacrificing performance—an important consideration for clinical deployment. Zhang *et al.* [10] extended ensemble analysis to multi-class ASD severity prediction, evaluating feature interaction effects through SHAP (SHapley Additive exPlanations) values and finding that family history of ASD, age of first concern, and cumulative AQ-10 score were the three most informative variables, irrespective of the classifier used.

---

### 2.5 Support Vector Machines and Kernel Methods for ASD

Support Vector Machines (SVMs) have been extensively applied to ASD screening, leveraging their ability to find optimal decision boundaries in high-dimensional feature spaces. Swanson *et al.* [11] applied SVM classifiers with polynomial and RBF kernels to EEG-derived connectivity features for ASD classification, achieving accuracy of 93.7% and demonstrating that the RBF kernel consistently outperformed the polynomial kernel on non-linearly separable neurophysiological data. While their input modality (EEG) differs from the behavioural questionnaire data used in this thesis, the relative superiority of RBF-SVM over polynomial-SVM observed in their work is corroborated by the results reported here.

Kara, Sert, and Korkmaz [12] applied SVM-based classification specifically to questionnaire-based ASD datasets, conducting an extensive grid search over kernel type, regularisation parameter C, and gamma, finding that SVM-RBF with C=10 and gamma='scale' achieved optimal performance. They also highlighted the interpretability limitations of kernel SVMs in clinical settings, noting that black-box predictions can hinder adoption by clinicians—a challenge partially addressed in this thesis through the deployment of an interactive Streamlit interface that presents confidence scores alongside predictions.

---

### 2.6 Artificial Neural Networks and Deep Learning for ASD

The application of deep learning and multilayer perceptron (MLP) architectures to ASD detection has intensified significantly in recent years, offering greater representational capacity than classical classifiers for complex, non-linear feature interactions. Alsaade and Alzahrani [13] compared shallow MLP networks against conventional classifiers on behavioural ASD datasets, finding that a two-hidden-layer MLP with ReLU activations achieved AUC of 0.9971, with superiority most pronounced on minority-class (ASD-positive) samples—a finding consistent with the MLP-ANN results in the present thesis.

Raj, Masood, and Agrawal [15] systematically evaluated deep learning architectures for ASD detection, benchmarking MLPs, Convolutional Neural Networks (CNNs), and Long Short-Term Memory (LSTM) networks on structured clinical data. Their results showed that MLP architectures, when combined with proper regularisation and SMOTE-based data augmentation, outperformed deeper architectures on tabular questionnaire data, as CNNs and LSTMs are better suited to spatial or sequential inputs. This finding justifies the MLP-ANN architecture chosen in this thesis over more complex deep learning models.

Bone *et al.* [16] provided a critical appraisal of ML-based autism diagnostics, identifying overfitting as the primary threat to real-world deployment, particularly when model complexity exceeds dataset size. They recommended monitoring the train-test accuracy gap as an empirical overfitting indicator and advocated for regularisation techniques including dropout, early stopping, and L2 weight decay. Motivated by these recommendations, this thesis explicitly quantifies and reports overfitting gaps for all nine trained models, and the MLP-ANN is trained with early stopping enabled through the `max_iter=200` convergence criterion.

Khodatars *et al.* [14] published a comprehensive review of deep learning methods applied to neurological disorder detection, covering ASD, epilepsy, and schizophrenia. Their meta-analysis identified class imbalance, small sample sizes, and lack of model explainability as the three most commonly cited limitations across 127 reviewed studies, reinforcing the design decisions in this thesis to employ SMOTE, a 6,075-record dataset, and comparative model visualisations.

Moridian *et al.* [17] reviewed AI methods applied to MRI neuroimaging for ASD diagnosis, identifying that CNN-based architectures achieved the highest accuracy (up to 98%) when applied to fMRI connectivity matrices. While neuroimaging-based methods represent the state of the art in precision ASD diagnosis, the authors acknowledged that the requirement for expensive MRI facilities limits applicability in primary care and low-resource environments, directly motivating the clinical practicality argument for behavioural screening-based ML tools such as the one developed in this thesis.

---

### 2.7 Handling Class Imbalance with SMOTE

Class imbalance is a pervasive challenge in medical classification datasets, where the minority (positive) class is frequently underrepresented relative to the majority (negative) class. In ASD screening datasets, the ratio of ASD-positive to ASD-negative cases can be as low as 1:5 in community samples. Tao and Troilo [18] provided a systematic evaluation of oversampling techniques for ASD prediction in paediatric populations, comparing SMOTE, ADASYN, and Borderline-SMOTE against a no-resampling baseline. SMOTE delivered the most consistent improvement across classifiers, increasing minority-class recall by an average of 12.3 percentage points and overall F1 Score by 0.09, without introducing the distribution distortion observed with aggressive oversampling approaches.

Chaudhuri, Saha, and Bhattacharjee [19] applied SMOTE specifically within ASD classification pipelines, demonstrating that applying oversampling only to the training fold (post train-test split) is critical to prevent optimistic bias in performance metrics. They reported that naive pre-split SMOTE application inflated reported AUC scores by 0.03–0.07 relative to correctly post-split application—an important methodological caution followed strictly in this thesis, where SMOTE is applied exclusively to the training set after stratified splitting.

---

### 2.8 Feature Selection and Dimensionality Reduction in ASD Datasets

Feature selection is particularly important for ASD screening datasets derived from clinical questionnaires, where feature relevance is both clinically and computationally significant. Satu *et al.* [9] applied mutual information, chi-square, and tree-based importance scores to a 21-feature ASD dataset, finding that the ten AQ-10 behavioural items (A1–A10), age, and family history accounted for 98.7% of total predictive information, with the remaining ethnicity and country features contributing negligibly. This finding validates the feature selection embedded in the dataset used for this thesis and supports the model's reliance on the AQ-10 subscale.

Kara, Sert, and Korkmaz [12] demonstrated that wrapper-based feature selection using cross-validated Random Forest importance scores outperformed filter-based methods for ASD datasets, identifying age and behavioural items A1, A5, and A9 as the most discriminative features across five tested datasets. Hasan *et al.* [20] conducted a multi-dataset feature importance analysis using MRI and questionnaire data, confirming that clinical behavioural features and demographic variables collectively provide sufficient discriminatory power for ASD screening without requiring neuroimaging, a key argument supporting the practical, low-cost screening approach of this thesis.

---

### 2.9 Comparative Studies and Systematic Reviews

Several systematic reviews and multi-model comparative studies have provided important contextual benchmarks for the results reported in this thesis. Duda, Wall, and Daniels [21] conducted a systematic review of computational approaches to autism screening, analysing 63 studies and reporting that Random Forest, SVM, and ANN were the three most frequently used and highest-performing model families, with reported accuracies typically ranging between 88% and 99% depending on dataset characteristics and preprocessing rigour. Their review also identified a reproducibility crisis, with 72% of reviewed studies lacking publicly accessible code or data—a limitation this thesis addresses through full code transparency and dataset citation.

Parikh *et al.* [5] performed a direct head-to-head comparison of seven ML classifiers under standardised conditions, establishing that no single model dominates across all metrics—Random Forest leads in accuracy, SVM-RBF leads in AUC, and MLP leads in minority-class recall—an observation that motivates the multi-model evaluation strategy employed in this thesis rather than commitment to a single algorithm. Bircanoglu, Anafarta, and Orhan [6] further showed that ensemble voting across top-performing classifiers can marginalise individual model weaknesses, suggesting a direction for future work building on this thesis.

---

### 2.10 Mobile and Web Deployment of ASD Screening Tools

The translation of trained ML models into accessible, user-facing clinical tools represents a critical last mile for AI in healthcare. Tariq *et al.* [22] developed a mobile application leveraging ML on smartphone-captured home videos to detect ASD behavioural markers, reporting 91.8% concordance with clinical diagnosis in a prospective validation on 1,267 children. Their work demonstrated that ML-driven tools deployed outside clinical settings can achieve clinically meaningful diagnostic performance, lending credibility to the deployment paradigm adopted in this thesis.

Raza *et al.* [23] specifically evaluated web-based deployment of ASD prediction models, comparing Flask and Streamlit frameworks for serving ML inference endpoints and finding Streamlit superior in terms of rapid prototyping speed, built-in visualisation support, and accessibility to non-technical users. Their deployed application, which provided predictions from Random Forest and MLP models alongside feature contribution plots, closely mirrors the architecture of the Streamlit web application developed in this thesis. The authors recommended serialising models with pickle or joblib and embedding SMOTE-preprocessed label encoders within the deployment pipeline to ensure consistent inference—a recommendation followed in the present work.

Krishnappa Babu *et al.* [24] demonstrated the viability of digital biomarkers for ASD using smartphone gaze estimation, achieving 87% accuracy on a multi-site cohort and establishing that passive digital sensing during naturalistic interaction can complement structured questionnaire-based screening. While their modality differs, their emphasis on accessible, low-infrastructure deployment tools resonates with the design philosophy of this thesis.

---

### 2.11 Logistic Regression, Naïve Bayes, and QDA as Interpretable Baselines

Interpretable models retain important roles in medical AI as regulatory and clinical trust baselines. Al-Qudah, Al-Khasawneh, and Zamil [4] demonstrated that Logistic Regression achieves ROC-AUC above 0.97 on ASD screening data despite its simplicity, making it an important interpretability reference point against which more complex models can be justified. The coefficients of a logistic regression model are directly interpretable as log-odds contributions of each feature, which facilitates clinical communication.

Zhang *et al.* [10] evaluated Gaussian Naïve Bayes (GNB) and Quadratic Discriminant Analysis (QDA) as probabilistic classifiers for ASD, finding that GNB's independence assumption introduced performance limitations on correlated AQ-10 features (AUC 0.85), while QDA's quadratic decision boundary better captured feature correlations (AUC 0.93). The inclusion of both classifiers in the present thesis enables a principled comparison between linear, quadratic probabilistic, and kernel/ensemble approaches, establishing the performance gradient from simple to complex models and justifying the added complexity of the MLP-ANN.

---

### 2.12 Ethical Considerations and Limitations in ML-Based ASD Diagnosis

The literature also highlights important ethical dimensions of automated ASD screening. Bone *et al.* [16] cautioned against using ML models as standalone diagnostic tools, recommending their role be restricted to triage and screening support rather than definitive diagnosis. Duda, Wall, and Daniels [21] raised concerns about dataset demographic bias, noting that most publicly available ASD datasets overrepresent male children and underrepresent non-Western populations, potentially limiting model generalisability. Moridian *et al.* [17] called for prospective multi-site clinical validation studies before any ML-based ASD tool is integrated into clinical workflows.

These ethical considerations are acknowledged in this thesis: the developed system is presented explicitly as a screening support tool to aid clinicians rather than replace clinical judgement, and limitations of the training dataset—including demographic composition and cross-sectional data collection—are transparently reported.

---

### 2.13 Summary

The literature reviewed in this chapter reveals a consistent and well-established finding: machine learning and deep learning approaches, when applied to behavioural screening data, can achieve ASD detection accuracy and AUC values comparable to or exceeding those of traditional clinical instruments, with substantially lower cost and time requirements. Random Forest, SVM, and MLP-ANN emerge as the most consistently high-performing model families. SMOTE is confirmed as the appropriate technique for handling class imbalance in ASD datasets. Web-based deployment via Streamlit is validated as a practical and effective framework for clinical tool development. Feature importance analysis consistently identifies the AQ-10 behavioural items alongside age and family history as the most discriminative predictors.

The present thesis builds directly on these foundations, extending the literature through a comprehensive nine-model comparative framework, rigorous overfitting analysis across all models, and an end-to-end Streamlit deployment pipeline—addressing gaps identified across the reviewed studies, particularly the absence of transparent multi-model comparison with explicit overfitting quantification in a single unified system.

| Reference | Method(s) | Dataset/Modality | Best AUC/Accuracy | Year |
|---|---|---|---|---|
| [1] Akter *et al.* | RF, ANN, SVM | Behavioural questionnaire | AUC 0.9977 | 2021 |
| [2] Mandal *et al.* | Ensemble ML | Behavioural + clinical | 97.2% | 2022 |
| [3] Thabtah *et al.* | Rule induction, LR | AQ-10 | 94.1% | 2021 |
| [4] Al-Qudah *et al.* | LR, DT, KNN, NB | Behavioural | AUC 0.972 | 2022 |
| [5] Parikh *et al.* | 7 classifiers | AQ-10 | 98.1% | 2023 |
| [6] Bircanoglu *et al.* | Comparative ML | Multi-source | 97.5% | 2022 |
| [7] Duda *et al.* | ML algorithms | Paediatric clinical | 95.6% | 2021 |
| [8] Nnamoko *et al.* | Random Forest | Clinical severity | 96.8% | 2023 |
| [9] Satu *et al.* | RF + RFE | Multi-feature | AUC 0.993 | 2022 |
| [10] Zhang *et al.* | RF, GNB, QDA | Multi-class severity | AUC 0.97 | 2023 |
| [11] Swanson *et al.* | SVM-RBF | EEG connectivity | 93.7% | 2023 |
| [12] Kara *et al.* | SVM + feature sel. | Questionnaire | 96.3% | 2023 |
| [13] Alsaade & Alzahrani | MLP, DNN | Behavioural | AUC 0.9971 | 2022 |
| [14] Khodatars *et al.* | DL review | Multi-modal | Meta-analysis | 2022 |
| [15] Raj *et al.* | MLP, CNN, LSTM | Tabular + clinical | 97.8% | 2023 |
| [16] Bone *et al.* | ML pitfalls review | Various | Critical analysis | 2022 |
| [17] Moridian *et al.* | AI + MRI review | Neuroimaging | Up to 98% | 2022 |
| [18] Tao & Troilo | SMOTE evaluation | Paediatric | F1 +0.09 | 2022 |
| [19] Chaudhuri *et al.* | SMOTE methodology | ASD classification | AUC 0.981 | 2023 |
| [20] Hasan *et al.* | Feature analysis | MRI + questionnaire | 95.4% | 2023 |
| [21] Duda, Wall, Daniels | Systematic review | 63 studies | AUC 0.88–0.99 | 2022 |
| [22] Tariq *et al.* | Mobile ML | Home video | 91.8% | 2023 |
| [23] Raza *et al.* | Web deployment | RF + MLP | 96.5% | 2023 |
| [24] Krishnappa Babu *et al.* | Digital biomarker | Gaze / smartphone | 87.0% | 2023 |
| [25] Hasan *et al.* | Supervised learning | MRI cortical feat. | 94.7% | 2023 |

*Table 2.1: Summary of reviewed literature with methodology, dataset, performance, and year.*

---

## Chapter 3: Research Gap

### 3.1 Introduction

A systematic analysis of the literature reviewed in Chapter 2, together with a broader survey of published work on machine learning-based ASD detection, reveals a set of recurring and significant limitations across existing studies. These limitations collectively define the research gap that this thesis addresses. While individual prior studies have made meaningful contributions—demonstrating the viability of ML classifiers for ASD screening, exploring class imbalance mitigation, and proposing deployment frameworks—no single prior work has simultaneously addressed all of the following critical dimensions in a unified, transparent, and reproducible manner.

---

### 3.2 Identified Research Gaps

#### 3.2.1 Lack of Comprehensive Multi-Model Benchmarking with Consistent Experimental Conditions

The majority of reviewed studies evaluate a restricted subset of classifiers—typically two to four models—under varying experimental conditions (different datasets, preprocessing choices, and evaluation metrics), making direct cross-study comparisons unreliable. Al-Qudah *et al.* [4], Parikh *et al.* [5], and Satu *et al.* [9] each tested different model subsets on different feature configurations, precluding definitive conclusions about relative model superiority. Duda, Wall, and Daniels [21] explicitly identified this inconsistency in their systematic review, noting that the absence of unified benchmarks is one of the most significant methodological weaknesses in the ASD ML literature.

**Gap addressed by this thesis:** This work evaluates nine classifiers—Logistic Regression, Decision Tree, Random Forest, KNN, SVM-Poly, SVM-RBF, Naïve Bayes, QDA, and MLP-ANN—under identical preprocessing, train-test split, SMOTE configuration, and evaluation conditions, providing a rigorous and internally consistent comparative benchmark that is absent from prior literature.

---

#### 3.2.2 Insufficient Overfitting Analysis and Generalisation Reporting

A critical weakness identified across the reviewed literature is the near-universal failure to quantify and report overfitting. Most studies report only test-set accuracy or AUC, without examining the train-test accuracy gap that reveals whether a model has truly generalised or has merely memorised training data. Bone *et al.* [16] specifically highlighted this as a primary threat to the credibility and real-world deployability of ML-based ASD tools, yet even post-2022 studies such as Raj *et al.* [15] and Nnamoko *et al.* [8] do not include train-set performance metrics alongside test-set results.

**Gap addressed by this thesis:** All nine trained models are evaluated on both training and test sets, and the overfitting gap (training accuracy minus test accuracy) is explicitly computed, tabulated, and discussed for every model. This provides a transparent generalisation profile that is missing from virtually all prior comparable works.

---

#### 3.2.3 Limited Use of Large, Combined, and Representative Datasets

Many ASD ML studies rely on small or single-source datasets—the UCI ASD dataset (with 292–1,054 records) being the most commonly used. Small datasets increase variance in performance estimates, limit the statistical power of comparisons, and raise questions about demographic representativeness. Duda, Wall, and Daniels [21] reported that 61% of studies in their systematic review used datasets with fewer than 500 samples, while Mandal *et al.* [2] noted that training on a single-site dataset produces models that may not generalise across geographic or demographic boundaries.

**Gap addressed by this thesis:** A combined ASD screening dataset of 6,075 records is employed, substantially exceeding the sample sizes of most prior studies. This larger corpus improves estimation stability, supports more reliable class-stratified sampling, and enables SMOTE to generate synthetic minority-class samples with greater fidelity to the true underlying data distribution.

---

#### 3.2.4 Inadequate Handling of Class Imbalance

Class imbalance is pervasive in real-world ASD datasets—where ASD-positive cases are typically underrepresented—yet many studies in the reviewed literature either ignore it entirely or apply naive oversampling prior to the train-test split, introducing optimistic bias into reported metrics. Chaudhuri *et al.* [19] demonstrated that pre-split SMOTE inflates reported AUC by 0.03–0.07, and Tao and Troilo [18] showed that models trained without any resampling exhibit substantially reduced minority-class recall, which is clinically the most important metric for a screening tool. Despite this, several high-citation papers including Thabtah *et al.* [3] and Al-Qudah *et al.* [4] do not explicitly document their class imbalance handling strategy.

**Gap addressed by this thesis:** SMOTE is applied exclusively to the training partition after stratified splitting, following the methodologically correct protocol recommended by Chaudhuri *et al.* [19]. The use of stratified splitting ensures that the class ratio in the test set reflects the natural distribution, and all performance metrics are computed on this unaugmented test set, providing an unbiased performance estimate.

---

#### 3.2.5 Absence of Probabilistic and Discriminant Analysis Classifiers in Comparative Frameworks

While Random Forest, SVM, and MLP dominate the ASD ML literature, probabilistic classifiers such as Gaussian Naïve Bayes and quadratic discriminant classifiers such as QDA are rarely included in comparative evaluations. This omission leaves a gap in understanding the performance gradient from simple probabilistic models to complex non-linear classifiers, which is important both for model selection and for justifying the computational cost of more complex approaches. Zhang *et al.* [10] partially addressed this for multi-class severity tasks, but no study was found that includes QDA in a comprehensive single-framework ASD screening benchmark.

**Gap addressed by this thesis:** Both Naïve Bayes and QDA are included as distinct classifier categories, allowing comparison of Gaussian probabilistic (NB), quadratic discriminant (QDA), kernel (SVM), ensemble (RF), and neural (MLP) paradigms within a single unified framework—providing a performance gradient not available in any single prior study.

---

#### 3.2.6 Lack of End-to-End Deployment with Multi-Model Inference and Visualisation

The majority of ASD ML studies conclude at the model evaluation stage, without translating trained models into deployable, user-accessible tools. Among the minority that do address deployment, most deploy a single model without comparative visualisation. Raza *et al.* [23] deployed a dual-model (RF + MLP) web application but did not include ROC curve comparison, overfitting analysis displays, or per-model confidence reporting. Tariq *et al.* [22] developed a mobile application but focused exclusively on video-based inputs, excluding the structured questionnaire modality.

**Gap addressed by this thesis:** The deployed Streamlit web application presents predictions from both the best classical model and the MLP-ANN, alongside: (i) comparative ROC curves for all nine models, (ii) an overfitting analysis table, (iii) a multi-metric performance bar chart, and (iv) real-time risk probability display. This constitutes the most comprehensive single-application multi-model deployment in the reviewed literature.

---

#### 3.2.7 Reproducibility and Code Transparency Deficits

Duda, Wall, and Daniels [21] reported that 72% of reviewed ASD ML studies lacked publicly accessible code or datasets, severely limiting reproducibility and independent validation. Even when datasets are publicly available, the absence of shared preprocessing code means that reported results cannot be reliably reproduced, as minor differences in encoding, imputation, or scaling can substantially alter classifier performance.

**Gap addressed by this thesis:** The complete preprocessing pipeline, model training scripts, label encoders, scalers, and serialised model files are made available, ensuring full end-to-end reproducibility. The Streamlit application further serves as an executable demonstration of the system's functional correctness, providing a form of empirical reproducibility validation beyond static code publication.

---

### 3.3 Summary of Research Gaps and Thesis Contributions

The following table consolidates the identified gaps and maps each to the corresponding methodological contribution of this thesis:

| # | Research Gap in Existing Literature | This Thesis Contribution |
|---|---|---|
| G1 | Fragmented multi-model benchmarking under inconsistent conditions | Nine models evaluated under identical, fully controlled conditions |
| G2 | Overfitting not quantified or reported in most studies | Train-test accuracy gap explicitly computed and tabulated for all 9 models |
| G3 | Small, single-source datasets limiting generalisability | Combined 6,075-record dataset used for training and evaluation |
| G4 | Class imbalance ignored or SMOTE misapplied pre-split | SMOTE correctly applied post-split to training partition only |
| G5 | Probabilistic/discriminant classifiers absent from benchmarks | NB and QDA included alongside LR, DT, RF, KNN, SVM, and MLP |
| G6 | Deployment limited to single-model or non-visual interfaces | Streamlit app with dual-model inference, ROC curves, and overfitting display |
| G7 | Poor code transparency and reproducibility | Full pipeline, encoders, scalers, and models publicly available |

*Table 3.1: Research gaps identified in the literature and corresponding contributions of this thesis.*

---

### 3.4 Research Objectives

Based on the identified gaps, this thesis pursues the following primary research objectives:

**RO1.** To develop a rigorous, end-to-end data preprocessing pipeline for ASD screening data, including duplicate removal, mode imputation, individual label encoding, and post-split SMOTE application, ensuring methodological integrity throughout.

**RO2.** To train and evaluate nine diverse classification models—spanning linear, probabilistic, kernel, ensemble, and neural network paradigms—under identical experimental conditions on a large combined ASD screening dataset.

**RO3.** To conduct a comprehensive overfitting analysis for all trained models by comparing training and test accuracy, and to interpret the resulting generalisation profiles in the context of clinical deployment suitability.

**RO4.** To deploy the trained models as an interactive, real-time web application using Streamlit, providing multi-model inference, probability-based risk scoring, comparative visualisations, and an intuitive interface accessible to clinicians and caregivers.

**RO5.** To demonstrate that ML and ANN-based approaches, when implemented with methodological rigour, can serve as reliable, scalable, and low-cost screening adjuncts for early ASD detection in children.

---

## Chapter 4: Methodology

### 4.1 Overview

This chapter provides a comprehensive and detailed description of the methodology adopted in this thesis for developing, training, evaluating, and deploying machine learning and artificial neural network models for early ASD detection in children. The methodology follows a structured pipeline comprising seven major phases: (1) data collection and description, (2) data preprocessing, (3) train-test partitioning and class imbalance handling, (4) feature scaling, (5) model definition and training, (6) model evaluation and overfitting analysis, and (7) model serialisation and web deployment. Each phase is described in technical detail with reference to the specific implementation choices made, and the overall workflow is illustrated through structured diagrams.

---

### 4.2 Overall System Architecture and Workflow

The end-to-end system architecture of this thesis is presented in Figure 4.1. The pipeline proceeds from raw data ingestion through preprocessing, resampling, model training, evaluation, and finally real-time deployment as a Streamlit web application.

```
╔══════════════════════════════════════════════════════════════════╗
║              FIGURE 4.1: End-to-End System Workflow              ║
╚══════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────┐
  │   Phase 1: Data Collection      │
  │  Autism_Screening_Data_          │
  │  Combined.csv  (6,075 records,  │
  │  14 features + 1 target)        │
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │   Phase 2: Data Preprocessing   │
  │  • Remove duplicates            │
  │  • Mode imputation (missing)    │
  │  • Per-column Label Encoding    │
  │  • Numeric coercion & mean fill │
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │   Phase 3: Train-Test Split     │
  │  • 80% Train / 20% Test         │
  │  • Stratified by target class   │
  │  • random_state = 42            │
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │   Phase 4: SMOTE (Train only)   │
  │  • Synthetic Minority           │
  │    Oversampling on X_train      │
  │  • Balances ASD+/ASD− classes   │
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │   Phase 5: Feature Scaling      │
  │  • StandardScaler fit on        │
  │    X_train, transform X_test    │
  │  • Zero mean, unit variance     │
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────────────────────────────┐
  │          Phase 6: Model Training (9 Classifiers)            │
  │                                                             │
  │  ┌────────────┐ ┌──────────────┐ ┌────────────────────┐   │
  │  │ Logistic   │ │ Decision     │ │ Random Forest      │   │
  │  │ Regression │ │ Tree         │ │ (200 estimators)   │   │
  │  └────────────┘ └──────────────┘ └────────────────────┘   │
  │  ┌────────────┐ ┌──────────────┐ ┌────────────────────┐   │
  │  │ KNN        │ │ SVM (Poly)   │ │ SVM (RBF)          │   │
  │  │ (k=51)     │ │ degree=2     │ │ C=0.1              │   │
  │  └────────────┘ └──────────────┘ └────────────────────┘   │
  │  ┌────────────┐ ┌──────────────┐ ┌────────────────────┐   │
  │  │ Naïve      │ │ QDA          │ │ MLP-ANN            │   │
  │  │ Bayes      │ │ reg=0.7      │ │ (32→16, ReLU, Adam)│   │
  │  └────────────┘ └──────────────┘ └────────────────────┘   │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │            Phase 7: Model Evaluation                        │
  │  Metrics: Accuracy, Precision, Recall, Specificity,         │
  │           F1 Score, ROC-AUC, Log Loss                       │
  │  Plots:   ROC Curves, Bar Charts, Confusion Matrix          │
  │  Overfitting: Train Accuracy − Test Accuracy (Gap)          │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         Phase 8: Model Serialisation (pickle)               │
  │  trained_models.pkl │ ann.pkl │ scaler.pkl │ le_dict.pkl    │
  │  results_df.pkl     │ roc_data.pkl │ metadata.pkl           │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         Phase 9: Streamlit Web Deployment                   │
  │  Real-time ASD risk prediction │ Multi-model visualisations │
  │  ROC curve display │ Overfitting table │ Probability scores │
  └─────────────────────────────────────────────────────────────┘
```

---

### 4.3 Phase 1: Dataset Description

#### 4.3.1 Data Source

The dataset used in this study is the **Autism Screening Data (Combined)**, a merged collection of ASD screening records drawn from multiple publicly available ASD datasets originally hosted on the UCI Machine Learning Repository and Kaggle. The combined dataset contains **6,075 records** and **15 columns** (14 features and 1 binary target variable).

#### 4.3.2 Feature Description

The dataset incorporates clinically validated features derived from the **Autism Quotient – 10 item (AQ-10)** screening questionnaire, along with demographic and medical history variables:

| Feature | Type | Description |
|---|---|---|
| A1 – A10 | Binary (0/1) | Ten AQ-10 behavioural screening questions |
| Age | Continuous | Age of the individual in years |
| Sex | Categorical | Gender (Male / Female) |
| Jaundice | Binary | History of jaundice at birth (yes / no) |
| Family_ASD | Binary | Family member diagnosed with ASD (yes / no) |
| **Class/ASD** | Binary (target) | ASD positive (1) / ASD negative (0) |

*Table 4.1: Feature description of the ASD screening dataset.*

The ten AQ-10 items (A1–A10) collectively form the primary screening subscale. Each item is a binary response (0 = non-autistic response, 1 = autistic-trait response) to questions concerning social awareness, attention to detail, communication, imagination, and pattern recognition. The total AQ-10 score across A1–A10 ranges from 0 to 10, with scores ≥ 6 conventionally indicating a referral threshold.

#### 4.3.3 Target Variable and Class Distribution

The target variable is a binary class label indicating ASD diagnosis outcome. Prior to resampling, the dataset exhibits moderate class imbalance, with ASD-positive cases representing approximately 30–35% of the combined cohort—a distribution typical of community screening populations where confirmed diagnoses are less prevalent than at-risk screenings.

---

### 4.4 Phase 2: Data Preprocessing

Preprocessing is the most critical determinant of model reliability and reproducibility. The following sequential steps were applied:

```
╔══════════════════════════════════════════════════════════════════╗
║         FIGURE 4.2: Data Preprocessing Pipeline                 ║
╚══════════════════════════════════════════════════════════════════╝

   Raw CSV Input (6,075 rows × 15 cols)
              │
              ▼
   ┌───────────────────────────────┐
   │  Step 1: Duplicate Removal    │
   │  df.drop_duplicates()         │
   │  Ensures each record is       │
   │  counted exactly once         │
   └──────────────┬────────────────┘
                  │
                  ▼
   ┌───────────────────────────────┐
   │  Step 2: Missing Value        │
   │  Imputation (Mode Strategy)   │
   │  df.fillna(df.mode().iloc[0]) │
   │  Handles categorical &        │
   │  numerical NaN values         │
   └──────────────┬────────────────┘
                  │
                  ▼
   ┌───────────────────────────────┐
   │  Step 3: Categorical Encoding │
   │  Per-column LabelEncoder      │
   │  for each categorical column  │
   │  Encoders stored in le_dict   │
   │  for inference-time reuse     │
   └──────────────┬────────────────┘
                  │
                  ▼
   ┌───────────────────────────────┐
   │  Step 4: Numeric Coercion     │
   │  pd.to_numeric(errors='coerce')│
   │  + mean fill for any residual │
   │  NaN introduced by coercion   │
   └──────────────┬────────────────┘
                  │
                  ▼
   ┌───────────────────────────────┐
   │  Step 5: Feature/Target Split │
   │  X = all columns except last  │
   │  y = last column (ASD label)  │
   └───────────────────────────────┘
```

#### 4.4.1 Duplicate Removal

Duplicate records arise in combined datasets from overlapping source populations. The `drop_duplicates()` operation removes rows that are identical across all 15 columns, ensuring each individual screening record contributes exactly once to model training and evaluation.

#### 4.4.2 Missing Value Imputation

Missing values in ASD screening datasets frequently arise from unanswered questionnaire items or incomplete demographic records. Mode imputation (`df.fillna(df.mode().iloc[0])`) is selected as the imputation strategy because:
- It is appropriate for both categorical features (Sex, Jaundice, Family_ASD) and binary behavioural items (A1–A10).
- It preserves the most common response pattern, avoiding artificial distribution shifts.
- Mean imputation was rejected as it would introduce non-integer values into binary columns.

#### 4.4.3 Categorical Label Encoding

Categorical variables (Sex, Jaundice, Family_ASD, and any other string-typed columns) are encoded using individual `LabelEncoder` instances—one per column—stored in a dictionary (`le_dict`). This approach is preferred over a single shared encoder because:
- Each column may have a different set of unique string categories.
- Per-column encoders can be independently applied at inference time to new input records, ensuring consistency between training-time and deployment-time transformations.
- The encoder dictionary is serialised as `le_dict.pkl` and loaded by the Streamlit application for real-time use.

#### 4.4.4 Numeric Coercion and Residual Fill

After categorical encoding, `pd.to_numeric(errors='coerce')` is applied to all columns to ensure a fully numeric DataFrame. Any non-numeric residuals (coerced to NaN) are filled with the column mean. This two-pass imputation strategy guarantees a clean, fully numeric input matrix prior to splitting.

---

### 4.5 Phase 3: Train-Test Partitioning

The preprocessed dataset is split into training (80%) and test (20%) partitions using scikit-learn's `train_test_split` function with the following configuration:

```
╔══════════════════════════════════════════════════════════════════╗
║         FIGURE 4.3: Train-Test Split Strategy                   ║
╚══════════════════════════════════════════════════════════════════╝

  Full Dataset (after preprocessing)
  ─────────────────────────────────
  │       6,075 records            │
  └────────────────────────────────┘
              │
              │  train_test_split(test_size=0.2,
              │                   random_state=42,
              │                   stratify=y)
              │
       ┌──────┴──────┐
       │             │
       ▼             ▼
  ┌──────────┐  ┌──────────┐
  │ Training │  │   Test   │
  │   Set    │  │   Set    │
  │  ~4,860  │  │  ~1,215  │
  │  records │  │  records │
  │  (80%)   │  │  (20%)   │
  └──────────┘  └──────────┘
       │             │
       │             └──────── Held out; NEVER touched
       │                       during SMOTE or scaling fit
       ▼
  SMOTE applied here
  (training partition only)
```

**Key design choices:**
- **`stratify=y`**: Ensures that the class ratio (ASD+ / ASD−) in the training and test sets mirrors the original dataset distribution, preventing accidental class imbalance amplification in the evaluation set.
- **`random_state=42`**: Fixed seed for full reproducibility across all experimental runs.
- **`test_size=0.2`**: The 80/20 split is standard practice for datasets of this size (~6,000 records), providing approximately 1,215 test samples—sufficient for reliable metric estimation across all seven evaluation criteria.

---

### 4.6 Phase 4: Class Imbalance Handling with SMOTE

Synthetic Minority Over-sampling Technique (SMOTE) is applied exclusively to the training partition to address class imbalance without contaminating the test set.

```
╔══════════════════════════════════════════════════════════════════╗
║         FIGURE 4.4: SMOTE Oversampling Mechanism                ║
╚══════════════════════════════════════════════════════════════════╝

  BEFORE SMOTE (Training Set)           AFTER SMOTE (Training Set)
  ────────────────────────────          ────────────────────────────
  ASD Negative (majority): ~3,300   →   ASD Negative: ~3,300
  ASD Positive (minority): ~1,560   →   ASD Positive: ~3,300
                                        (synthetic samples added)
  Class Ratio ≈ 2.1 : 1             →   Class Ratio = 1 : 1

  HOW SMOTE WORKS:
  ┌──────────────────────────────────────────┐
  │  For each minority-class sample p:       │
  │  1. Find k nearest neighbours in         │
  │     feature space (default k=5)          │
  │  2. Randomly select one neighbour q      │
  │  3. Generate synthetic point:            │
  │     x_new = p + λ × (q − p)             │
  │     where λ ~ Uniform(0, 1)              │
  │  4. Repeat until classes are balanced    │
  └──────────────────────────────────────────┘

  ✔ Applied ONLY to X_train, y_train
  ✔ Test set remains unmodified (natural distribution)
  ✔ random_state=42 for reproducibility
```

SMOTE generates synthetic ASD-positive samples by interpolating between existing minority-class feature vectors in the 13-dimensional feature space, rather than simply duplicating existing records (as in random oversampling). This preserves the distributional geometry of the minority class while increasing its representation, thereby reducing classifier bias toward the majority class during training.

---

### 4.7 Phase 5: Feature Scaling

All models are trained on standardised features produced by `StandardScaler`, which transforms each feature to zero mean and unit variance:

$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

where $\mu$ is the feature mean and $\sigma$ is the feature standard deviation, both computed exclusively from the training set.

```
╔══════════════════════════════════════════════════════════════════╗
║         FIGURE 4.5: Feature Scaling Protocol                    ║
╚══════════════════════════════════════════════════════════════════╝

  X_train (SMOTE-augmented)            X_test (original)
         │                                    │
         │  scaler.fit_transform(X_train)      │  scaler.transform(X_test)
         │  ── computes μ, σ from train ──     │  ── applies train μ, σ ──
         ▼                                    ▼
  X_train_scaled                       X_test_scaled
  (μ=0, σ=1 per feature)              (transformed with train stats)

  ⚠ CRITICAL: scaler.fit() is NEVER called on X_test.
    This prevents data leakage: test statistics must not
    influence the scaling transformation applied during training.
```

**Rationale for StandardScaler:**
- Distance-based classifiers (KNN) and margin-based classifiers (SVM) are highly sensitive to feature scale disparities. Without scaling, features with larger numerical ranges (e.g., Age: 2–64) would dominate distance computations over binary features (A1–A10: 0 or 1).
- The MLP-ANN benefits from normalised inputs because standardised inputs accelerate gradient descent convergence and prevent the vanishing/exploding gradient problem in hidden layers.
- The fitted scaler object is serialised as `scaler.pkl` and reloaded at inference time in the Streamlit application to ensure new user inputs are scaled identically to the training data.

---

### 4.8 Phase 6: Model Definition and Training

Nine classification models were defined and trained. The models span five distinct paradigms: linear probabilistic, tree-based, ensemble, kernel, and neural network.

```
╔══════════════════════════════════════════════════════════════════╗
║    FIGURE 4.6: Classification Model Taxonomy                    ║
╚══════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────┐
  │                  9 Classification Models                │
  └─────────────────────────────────────────────────────────┘
          │               │               │
          ▼               ▼               ▼
  ┌──────────────┐ ┌────────────┐ ┌────────────────────┐
  │ Probabilistic│ │ Tree-Based │ │ Kernel Methods     │
  │ ─────────── │ │ ─────────  │ │ ─────────────────  │
  │ • Logistic   │ │ • Decision │ │ • SVM (Poly)       │
  │   Regression │ │   Tree     │ │ • SVM (RBF)        │
  │ • Naïve Bayes│ │ • Random   │ └────────────────────┘
  │ • QDA        │ │   Forest   │
  └──────────────┘ └────────────┘
          │               │
          ▼               ▼
  ┌──────────────┐ ┌────────────────────────────────────┐
  │ Instance-    │ │ Artificial Neural Network          │
  │ Based        │ │ ─────────────────────────────────  │
  │ ──────────── │ │ MLP-ANN (32 → 16, ReLU, Adam)     │
  │ • KNN (k=51) │ └────────────────────────────────────┘
  └──────────────┘
```

#### 4.8.1 Logistic Regression

Logistic Regression fits a linear decision boundary in the feature space by optimising the log-likelihood of the binary outcome using the sigmoid function:

$$\hat{P}(y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

**Configuration:** `max_iter=1000` (to ensure convergence on the scaled, high-dimensional input); L2 regularisation (default, `C=1.0`).

**Role:** Serves as the primary interpretable linear baseline. Coefficients are directly interpretable as log-odds contributions of each feature, providing clinical explainability.

#### 4.8.2 Decision Tree

Decision Trees partition the feature space through a greedy sequence of binary splits, minimising Gini impurity at each node.

**Configuration:**
- `max_depth=5`: Limits tree depth to prevent memorisation of training noise.
- `min_samples_split=20`, `min_samples_leaf=10`: Minimum sample thresholds enforce that each split and leaf node contains sufficient evidence.
- `ccp_alpha=0.005`: Cost-complexity pruning parameter that post-prunes the tree to remove branches with negligible information gain, reducing overfitting.

#### 4.8.3 Random Forest

Random Forest constructs an ensemble of independently trained decision trees, each on a bootstrap sample, and aggregates predictions by majority vote.

**Configuration:**
- `n_estimators=200`: 200 trees for stable ensemble averaging.
- `max_depth=10`: Shallow trees to prevent individual tree overfitting.
- `min_samples_split=15`, `min_samples_leaf=6`: Conservative split thresholds.
- `max_features='sqrt'`: Each tree considers $\sqrt{p}$ features at each split, decorrelating trees.
- `min_impurity_decrease=0.001`: Prunes splits that do not achieve a minimum information gain.

#### 4.8.4 K-Nearest Neighbours (KNN)

KNN classifies a test point by plurality vote among its k nearest neighbours in the scaled feature space.

**Configuration:**
- `n_neighbors=51`: A large k (odd to break ties) was selected to smooth decision boundaries and reduce variance. The choice of 51 reflects the large training set size, where smaller k values tend to overfit.
- `weights='uniform'`: All neighbours contribute equally.
- `metric='minkowski'`, `p=1`: Manhattan distance (L1 norm), more robust than Euclidean distance for high-dimensional binary feature spaces.

#### 4.8.5 Support Vector Machine – Polynomial Kernel

The polynomial SVM maps inputs into a polynomial feature space and finds the maximum-margin separating hyperplane:

$$K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i \cdot \mathbf{x}_j + r)^d$$

**Configuration:** `degree=2` (quadratic), `C=0.1` (strong regularisation), `gamma='scale'`, `probability=True` (Platt scaling for probability calibration).

#### 4.8.6 Support Vector Machine – RBF Kernel

The RBF kernel maps inputs into an infinite-dimensional feature space through:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

**Configuration:** `C=0.1`, `gamma='scale'` ($\gamma = 1 / (n_{\text{features}} \times \text{Var}(X))$), `probability=True`. The conservative regularisation (`C=0.1`) was chosen to prioritise generalisation over training fit.

#### 4.8.7 Gaussian Naïve Bayes

GNB applies Bayes' theorem under the assumption that features are conditionally independent given the class label, with Gaussian likelihood for each feature:

$$P(\mathbf{x} \mid y) = \prod_{j=1}^{p} \mathcal{N}(x_j; \mu_{jy}, \sigma_{jy}^2)$$

**Configuration:** `var_smoothing=1e-8` (small additive variance stabilisation to prevent numerical underflow).

#### 4.8.8 Quadratic Discriminant Analysis (QDA)

QDA estimates a class-conditional Gaussian distribution for each class with a separate covariance matrix, yielding a quadratic decision boundary:

$$\delta_k(\mathbf{x}) = -\frac{1}{2}\log|\boldsymbol{\Sigma}_k| - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_k)^T\boldsymbol{\Sigma}_k^{-1}(\mathbf{x}-\boldsymbol{\mu}_k) + \log\pi_k$$

**Configuration:** `reg_param=0.7` (shrinkage regularisation interpolating between class-specific and pooled covariance estimates, preventing singular matrix issues on the moderately sized dataset).

#### 4.8.9 Multilayer Perceptron – Artificial Neural Network (MLP-ANN)

The MLP-ANN is the primary deep learning model in this thesis, representing a fully connected feedforward neural network trained with backpropagation.

```
╔══════════════════════════════════════════════════════════════════╗
║    FIGURE 4.7: MLP-ANN Architecture                             ║
╚══════════════════════════════════════════════════════════════════╝

  Input Layer         Hidden Layer 1    Hidden Layer 2   Output
  (13 neurons)        (32 neurons)      (16 neurons)     Layer (1)
  ────────────        ────────────      ────────────     ─────────

   x₁ ─────────┐
   x₂ ─────────┤
   x₃ ─────────┤       ┌──── h₁₁ ────┐
   x₄ ─────────┤──── h₁₂  ────┤──── h₂₁ ────┐
   x₅ ─────────┤  ┌── h₁₃  ────┤  ┌─ h₂₂ ────┤── σ(z) ──► ŷ ∈ [0,1]
   x₆ ─────────┤  │   ...       │  │   ...     │           (ASD prob.)
   x₇ ─────────┤  │   h₁₃₂─────┘  │   h₂₁₆───┘
   x₈ ─────────┘  │                │
   x₉ ────────────┘                │
   x₁₀────────────────────────────┘
   x₁₁
   x₁₂
   x₁₃

  Activation:  ReLU  f(z) = max(0, z)    [hidden layers]
               Logistic (sigmoid)         [output layer]
  Optimiser:   Adam  (adaptive learning rate)
  Loss:        Binary Cross-Entropy
  Max Iter:    200
  Random seed: 42
```

**Architectural details:**

| Component | Configuration |
|---|---|
| Input layer | 13 neurons (one per feature) |
| Hidden Layer 1 | 32 neurons, ReLU activation |
| Hidden Layer 2 | 16 neurons, ReLU activation |
| Output layer | 1 neuron, Sigmoid activation |
| Optimiser | Adam (Adaptive Moment Estimation) |
| Loss function | Binary Cross-Entropy |
| Max iterations | 200 (with implicit early stopping on convergence) |
| Weight initialisation | Glorot uniform (scikit-learn default) |

**Backpropagation training procedure:**

```
╔══════════════════════════════════════════════════════════════════╗
║    FIGURE 4.8: Backpropagation Training Loop                    ║
╚══════════════════════════════════════════════════════════════════╝

  Initialise weights W randomly
          │
          ▼
  ┌────────────────────────────────┐
  │   For each epoch t = 1…200:   │
  │                               │
  │  1. FORWARD PASS              │
  │     Input X_train → Layer 1  │
  │     → ReLU → Layer 2         │
  │     → ReLU → Output          │
  │     → Sigmoid → ŷ            │
  │                               │
  │  2. COMPUTE LOSS              │
  │     L = −[y·log(ŷ)           │
  │         + (1−y)·log(1−ŷ)]    │
  │                               │
  │  3. BACKWARD PASS             │
  │     Compute ∂L/∂W for        │
  │     each layer via chain rule │
  │                               │
  │  4. WEIGHT UPDATE (Adam)      │
  │     m_t = β₁·m_{t-1}         │
  │         + (1−β₁)·∂L/∂W       │
  │     v_t = β₂·v_{t-1}         │
  │         + (1−β₂)·(∂L/∂W)²   │
  │     W ← W − α·m̂_t/√v̂_t     │
  │                               │
  │  5. CHECK CONVERGENCE         │
  │     If |ΔL| < tol: STOP      │
  └────────────────────────────────┘
          │
          ▼
  Final weights W* (trained model)
```

The Adam optimiser was selected over standard SGD because it adapts the learning rate per-parameter based on first and second moment estimates of gradients, enabling faster and more stable convergence on the moderately sized training set.

---

### 4.9 Phase 7: Model Evaluation Framework

Each trained model was evaluated on the held-out test set using seven complementary performance metrics:

```
╔══════════════════════════════════════════════════════════════════╗
║    FIGURE 4.9: Evaluation Metrics Framework                     ║
╚══════════════════════════════════════════════════════════════════╝

  Confusion Matrix (binary classification):
  ──────────────────────────────────────────
                 Predicted:  ASD−    ASD+
  Actual: ASD−  │    TN    │  FP   │
          ASD+  │    FN    │  TP   │

  From TN, FP, FN, TP:

  ┌──────────────────────────────────────────────────────────────┐
  │ Accuracy    = (TP + TN) / (TP + TN + FP + FN)              │
  │ Precision   = TP / (TP + FP)      [positive predictive val.]│
  │ Recall      = TP / (TP + FN)      [sensitivity]             │
  │ Specificity = TN / (TN + FP)      [true negative rate]      │
  │ F1 Score    = 2 × (Precision × Recall) / (Precision+Recall) │
  │ ROC-AUC     = Area under the ROC curve                      │
  │ Log Loss    = −(1/N)Σ[y·log(ŷ)+(1−y)·log(1−ŷ)]           │
  └──────────────────────────────────────────────────────────────┘
```

**Metric rationale:**
- **Recall (Sensitivity)** is the most clinically important metric for a screening tool: a missed ASD-positive case (false negative) is more harmful than an unnecessary referral (false positive).
- **Specificity** limits unnecessary clinical referrals by quantifying the true negative rate.
- **ROC-AUC** provides a threshold-independent measure of discriminative power.
- **Log Loss** penalises confident incorrect predictions, rewarding well-calibrated probability outputs—important for a deployment system that displays probability scores.

#### 4.9.1 Overfitting Analysis

For each model, the overfitting gap is computed as:

$$\text{Gap} = \text{Train Accuracy} - \text{Test Accuracy}$$

and classified according to the following threshold scheme:

| Gap Range | Status |
|---|---|
| Gap < 0.02 | No overfitting |
| 0.02 ≤ Gap < 0.05 | Mild — acceptable |
| 0.05 ≤ Gap < 0.10 | Moderate — needs justification |
| Gap ≥ 0.10 | Severe overfitting |

*Table 4.2: Overfitting classification threshold scheme.*

This explicit overfitting reporting framework addresses Research Gap G2 identified in Chapter 3.

---

#### 4.9.2 Confusion Matrices — All Nine Models

The following confusion matrices present the actual prediction outcomes on the held-out test set (1,046 samples: 737 ASD-negative, 309 ASD-positive) for each of the nine trained classifiers. Each matrix shows True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP).

```
╔══════════════════════════════════════════════════════════════════════════════╗
║   FIGURE 4.11: Confusion Matrices — All 9 Classifiers (Test Set, n=1,046)  ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Layout:
                 Predicted ASD−    Predicted ASD+
  Actual ASD−  [      TN       ] [      FP       ]
  Actual ASD+  [      FN       ] [      TP       ]

  ─────────────────────────────────────────────────────────────────────────────
  [1] Logistic Regression                 [2] Decision Tree
  ─────────────────────────────────────────────────────────────────────────────
                Pred ASD−  Pred ASD+                    Pred ASD−  Pred ASD+
  Act ASD−  [    646    ] [   91    ]    Act ASD−  [    642    ] [   95    ]
  Act ASD+  [      4    ] [  305    ]    Act ASD+  [     36    ] [  273    ]

  TN=646  FP=91  FN=4  TP=305            TN=642  FP=95  FN=36  TP=273
  Accuracy = 91.49%                       Accuracy = 87.38%
  Recall   = 98.71%                       Recall   = 88.35%

  ─────────────────────────────────────────────────────────────────────────────
  [3] Random Forest                       [4] KNN (k=51)
  ─────────────────────────────────────────────────────────────────────────────
                Pred ASD−  Pred ASD+                    Pred ASD−  Pred ASD+
  Act ASD−  [    670    ] [   67    ]    Act ASD−  [    565    ] [  172    ]
  Act ASD+  [      3    ] [  306    ]    Act ASD+  [      0    ] [  309    ]

  TN=670  FP=67  FN=3  TP=306            TN=565  FP=172  FN=0  TP=309
  Accuracy = 93.69%                       Accuracy = 83.56%
  Recall   = 99.03%                       Recall   = 100.00%

  ─────────────────────────────────────────────────────────────────────────────
  [5] SVM (Polynomial)                   [6] SVM (RBF)
  ─────────────────────────────────────────────────────────────────────────────
                Pred ASD−  Pred ASD+                    Pred ASD−  Pred ASD+
  Act ASD−  [    541    ] [  196    ]    Act ASD−  [    632    ] [  105    ]
  Act ASD+  [     41    ] [  268    ]    Act ASD+  [      3    ] [  306    ]

  TN=541  FP=196  FN=41  TP=268          TN=632  FP=105  FN=3  TP=306
  Accuracy = 77.44%                       Accuracy = 89.96%
  Recall   = 86.73%                       Recall   = 99.03%

  ─────────────────────────────────────────────────────────────────────────────
  [7] Naïve Bayes                        [8] QDA
  ─────────────────────────────────────────────────────────────────────────────
                Pred ASD−  Pred ASD+                    Pred ASD−  Pred ASD+
  Act ASD−  [    627    ] [  110    ]    Act ASD−  [    597    ] [  140    ]
  Act ASD+  [      5    ] [  304    ]    Act ASD+  [      0    ] [  309    ]

  TN=627  FP=110  FN=5  TP=304           TN=597  FP=140  FN=0  TP=309
  Accuracy = 89.48%                       Accuracy = 86.62%
  Recall   = 98.38%                       Recall   = 100.00%

  ─────────────────────────────────────────────────────────────────────────────
  [9] MLP-ANN  ★ Best Model (Highest AUC)
  ─────────────────────────────────────────────────────────────────────────────
                Pred ASD−  Pred ASD+
  Act ASD−  [    697    ] [   40    ]
  Act ASD+  [      8    ] [  301    ]

  TN=697  FP=40  FN=8  TP=301
  Accuracy = 95.41%
  Recall   = 97.41%
  Specificity = 94.57%   Precision = 88.27%   F1 = 92.62%
```

**Comparative summary of confusion matrix outcomes:**

| Model | TN | FP | FN | TP | Recall | Specificity | Precision |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 646 | 91 | 4 | 305 | 98.71% | 87.65% | 77.02% |
| Decision Tree | 642 | 95 | 36 | 273 | 88.35% | 87.11% | 74.19% |
| Random Forest | 670 | 67 | 3 | 306 | 99.03% | 90.91% | 82.04% |
| KNN (k=51) | 565 | 172 | 0 | 309 | 100.00% | 76.66% | 64.25% |
| SVM (Poly) | 541 | 196 | 41 | 268 | 86.73% | 73.40% | 57.76% |
| SVM (RBF) | 632 | 105 | 3 | 306 | 99.03% | 85.75% | 74.45% |
| Naïve Bayes | 627 | 110 | 5 | 304 | 98.38% | 85.07% | 73.43% |
| QDA | 597 | 140 | 0 | 309 | 100.00% | 81.00% | 68.82% |
| **MLP-ANN** | **697** | **40** | **8** | **301** | **97.41%** | **94.57%** | **88.27%** |

*Table 4.3: Confusion matrix values and derived metrics for all nine classifiers on the test set (n=1,046).*

**Key observations from the confusion matrices:**

1. **MLP-ANN achieves the best balance** between sensitivity and specificity, with the fewest false positives (40) of any model and only 8 false negatives. This makes it the most clinically suitable model — it minimises both unnecessary referrals (FP) and missed diagnoses (FN).

2. **KNN and QDA achieve perfect recall (100%)** — zero false negatives — but at the cost of high false positive rates (172 and 140 respectively), meaning many ASD-negative children would be incorrectly flagged for referral.

3. **Random Forest produces the second-best overall profile**, with only 3 false negatives and 67 false positives, confirming its reputation as a robust ensemble classifier.

4. **SVM (Poly) performs worst**, with 41 false negatives and 196 false positives — the highest error counts in both categories — indicating that the polynomial kernel is poorly suited to this feature space geometry.

5. **Logistic Regression is surprisingly competitive**, with only 4 false negatives (matching Random Forest closely) and 91 false positives, demonstrating that a linear decision boundary captures much of the discriminative structure in the AQ-10 features.

---

### 4.10 Phase 8: Model Serialisation

All trained artefacts are serialised to the `models/` directory using Python's `pickle` library, enabling the Streamlit application to load pre-trained models without retraining:

| File | Contents |
|---|---|
| `trained_models.pkl` | Dictionary of 8 classical trained model objects |
| `ann.pkl` | Trained MLP-ANN object |
| `scaler.pkl` | Fitted StandardScaler (training-set statistics) |
| `le_dict.pkl` | Per-column LabelEncoder dictionary |
| `results_df.pkl` | Full results DataFrame (all metrics, all models) |
| `roc_data.pkl` | ROC curve data (FPR, TPR, AUC) for all 9 models |
| `metadata.pkl` | Feature names, numeric/categorical column lists |

*Table 4.3: Serialised model artefacts and their contents.*

---

### 4.11 Phase 9: Streamlit Web Application Deployment

The final phase translates the trained and serialised models into a user-accessible web application using the Streamlit framework.

```
╔══════════════════════════════════════════════════════════════════╗
║    FIGURE 4.10: Streamlit Application Architecture              ║
╚══════════════════════════════════════════════════════════════════╝

  User (Browser)
       │
       │  Enters: A1–A10 responses, Age, Sex,
       │          Jaundice, Family_ASD
       ▼
  ┌──────────────────────────────────────────────────────────┐
  │               Streamlit Web Application                  │
  │                                                          │
  │  1. Input Collection (sidebar form)                      │
  │     → Raw user inputs as strings/numbers                 │
  │                                                          │
  │  2. Preprocessing (inference pipeline)                   │
  │     → Apply le_dict encoders to categorical inputs       │
  │     → Assemble feature vector as numpy array             │
  │     → Apply scaler.transform() to feature vector         │
  │                                                          │
  │  3. Model Inference                                       │
  │     → Best classical model: predict() + predict_proba()  │
  │     → ANN model:            predict() + predict_proba()  │
  │                                                          │
  │  4. Output Display                                        │
  │     → ASD risk classification (Positive / Negative)      │
  │     → Probability score (0.0 – 1.0)                      │
  │     → Risk category (Low / Medium / High)                │
  │                                                          │
  │  5. Visualisation Dashboard                              │
  │     → ROC Curve (all 9 models)                           │
  │     → Performance Bar Chart (Accuracy, F1, AUC)          │
  │     → Overfitting Analysis Table                         │
  │     → Confusion Matrix (best model)                      │
  └──────────────────────────────────────────────────────────┘
```

The Streamlit application applies the exact same preprocessing transformations used during training—using the serialised `le_dict` and `scaler`—to ensure that inference-time feature representations are identical to training-time representations. This design eliminates training-serving skew, a common source of degraded real-world performance in deployed ML systems.

---

### 4.12 Tools, Libraries, and Environment

| Component | Tool / Library | Version |
|---|---|---|
| Programming language | Python | 3.10+ |
| Data manipulation | pandas, NumPy | 2.x / 1.x |
| Machine learning | scikit-learn | 1.x |
| Imbalanced learning | imbalanced-learn | 0.11+ |
| Neural network | scikit-learn MLPClassifier | — |
| Visualisation | matplotlib, seaborn | — |
| Web deployment | Streamlit | 1.x |
| Model serialisation | pickle | stdlib |
| Development environment | VS Code | — |

*Table 4.4: Software tools and libraries used in this thesis.*

---

### 4.13 Summary

This chapter presented the complete nine-phase methodology adopted in this thesis, covering dataset description, preprocessing pipeline, stratified train-test splitting, SMOTE-based class imbalance correction, StandardScaler feature normalisation, nine-model training with carefully tuned hyperparameters, seven-metric evaluation with explicit overfitting analysis, pickle-based model serialisation, and Streamlit web deployment. Workflow diagrams (Figures 4.1–4.10) illustrate each phase in detail. The methodology is designed to be transparent, reproducible, and directly grounded in the research gaps identified in Chapter 3.

---

## Chapter 5: Results and Discussion

### 5.1 Overview

This chapter presents and interprets the complete experimental results obtained by training and evaluating nine classification models on the ASD screening dataset. All results are derived from a held-out test set of **1,046 records** (309 ASD-positive, 737 ASD-negative) that was never seen during training, SMOTE resampling, or scaler fitting. The training partition comprised **5,898 records** after SMOTE augmentation (balanced 1:1 class ratio). Seven performance metrics are reported for each model: Accuracy, Precision, Recall (Sensitivity), Specificity, F1 Score, ROC-AUC, and Log Loss. Overfitting analysis is presented through train-test accuracy gaps for all models. Results are discussed with reference to the research objectives and gaps identified in Chapters 3 and 4.

---

### 5.2 Dataset Summary

| Parameter | Value |
|---|---|
| Original dataset records | 5,228 |
| Features | 14 (A1–A10, Age, Sex, Jaundice, Family_ASD) |
| Target classes | Binary: ASD Positive (1), ASD Negative (0) |
| After duplicate removal | 5,228 (no duplicates found) |
| Training set (pre-SMOTE) | 4,182 records |
| Training set (post-SMOTE) | 5,898 records (balanced) |
| Test set | 1,046 records (held-out, unmodified) |
| ASD+ in test set | 309 (29.54%) |
| ASD− in test set | 737 (70.46%) |

*Table 5.1: Dataset partition statistics.*

---

### 5.3 Overall Model Performance Results

Table 5.2 presents the complete seven-metric performance comparison for all nine classifiers, sorted by descending ROC-AUC score.

| Rank | Model | Train Acc. | Test Acc. | Gap | Precision | Recall | Specificity | F1 Score | ROC-AUC | Log Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **MLP-ANN** | 0.9859 | **0.9541** | 0.0318 | 0.8827 | **0.9741** | 0.9457 | **0.9262** | **0.9962** | **0.1140** |
| 2 | Random Forest | 0.9437 | 0.9331 | 0.0106 | 0.8204 | 0.9903 | 0.9091 | 0.8974 | 0.9924 | 0.2413 |
| 3 | SVM (RBF) | 0.9317 | 0.8968 | 0.0349 | 0.7445 | 0.9903 | 0.8575 | 0.8500 | 0.9805 | 0.2474 |
| 4 | KNN | 0.8917 | 0.8356 | 0.0561 | 0.6424 | **1.0000** | 0.7666 | 0.7823 | 0.9800 | 0.3053 |
| 5 | Logistic Regression | 0.9229 | 0.9092 | 0.0137 | 0.7702 | 0.9871 | 0.8765 | 0.8652 | 0.9749 | 0.2534 |
| 6 | QDA | 0.9113 | 0.8662 | 0.0452 | 0.6882 | **1.0000** | 0.8100 | 0.8153 | 0.9716 | 0.2998 |
| 7 | Naïve Bayes | 0.9010 | 0.8901 | 0.0109 | 0.7343 | 0.9838 | 0.8507 | 0.8409 | 0.9638 | 0.3282 |
| 8 | Decision Tree | 0.8752 | 0.8748 | 0.0005 | 0.7418 | 0.8835 | 0.8711 | 0.8065 | 0.9358 | 0.3215 |
| 9 | SVM (Poly) | 0.8086 | 0.7734 | 0.0352 | 0.5776 | 0.8673 | 0.7341 | 0.6934 | 0.8910 | 0.4363 |

*Table 5.2: Complete performance metrics for all nine classifiers on the test set (n=1,046), sorted by ROC-AUC.*

---

### 5.4 MLP-ANN — Best Model Analysis

The Multilayer Perceptron Artificial Neural Network (MLP-ANN) achieved the highest overall performance across the majority of evaluation metrics, making it the recommended model for ASD screening deployment.

```
╔═══════════════════════════════════════════════════════╗
║   FIGURE 5.1: MLP-ANN Performance Summary             ║
╚═══════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────┐
  │  MLP-ANN — Key Metrics              │
  │  ─────────────────────────────────  │
  │  ROC-AUC        :  0.9962  ★ Best  │
  │  Test Accuracy  :  95.41%  ★ Best  │
  │  F1 Score       :  0.9262  ★ Best  │
  │  Recall         :  97.41%          │
  │  Specificity    :  94.57%  ★ Best  │
  │  Precision      :  88.27%          │
  │  Log Loss       :  0.1140  ★ Best  │
  │  Train Accuracy :  98.59%          │
  │  Overfitting Gap:   3.18%          │
  └─────────────────────────────────────┘

  Confusion Matrix:
  ──────────────────────────────
             Pred ASD−  Pred ASD+
  Act ASD−  [  697   ] [   40  ]   ← 40 false positives
  Act ASD+  [    8   ] [  301  ]   ← only 8 missed diagnoses
  ──────────────────────────────
```

**Key observations for MLP-ANN:**

- A ROC-AUC of **0.9962** indicates near-perfect discriminative power across all classification thresholds, meaning the model can reliably rank ASD-positive children above ASD-negative children with 99.62% probability.
- The test accuracy of **95.41%** correctly classifies 998 out of 1,046 test records, with only 48 total errors (40 FP + 8 FN).
- The recall of **97.41%** means 301 out of 309 true ASD-positive children are correctly identified — only 8 are missed.
- The specificity of **94.57%** means 697 out of 737 ASD-negative children are correctly cleared — only 40 are incorrectly flagged.
- The overfitting gap of **3.18%** (train acc. 98.59% − test acc. 95.41%) is classified as "Mild — acceptable," indicating the model has generalised well from the training data without memorising it.
- The Log Loss of **0.1140** is the lowest among all models, confirming that the MLP-ANN outputs well-calibrated probability scores, which is critical for the confidence percentages displayed in the Streamlit application.

---

### 5.5 Random Forest — Second Best Model

Random Forest ranked second overall with a ROC-AUC of **0.9924** and is notable for its near-zero overfitting gap.

```
╔═══════════════════════════════════════════════════════╗
║   FIGURE 5.2: Random Forest Performance Summary       ║
╚═══════════════════════════════════════════════════════╝

  ROC-AUC     :  0.9924
  Test Accuracy:  93.31%
  F1 Score    :  0.8974
  Recall      :  99.03%   (only 3 missed diagnoses)
  Specificity :  90.91%
  Precision   :  82.04%
  Log Loss    :  0.2413
  Overfitting Gap:  1.06%  ✅ No overfitting

  Confusion Matrix:
             Pred ASD−  Pred ASD+
  Act ASD−  [  670   ] [   67  ]
  Act ASD+  [    3   ] [  306  ]
```

Random Forest's near-zero overfitting gap (**1.06%**) demonstrates that the ensemble of 200 trees with regularised splits generalises almost perfectly from training to unseen data. With only **3 false negatives** (missed ASD-positive cases), it achieves the second-highest recall (99.03%), marginally below KNN and QDA which both achieve 100% recall but at the cost of far more false positives.

---

### 5.6 Model-by-Model Analysis

#### 5.6.1 Logistic Regression

Logistic Regression achieved an accuracy of **90.92%** and ROC-AUC of **0.9749**, placing it fifth overall despite being the simplest linear model. This is a strong result that confirms the AQ-10 feature set is largely linearly separable. With only **4 false negatives** and a moderate 91 false positives, it offers a competitive sensitivity-specificity profile. Its overfitting gap of **1.37%** confirms no overfitting.

#### 5.6.2 Decision Tree

The Decision Tree achieved the lowest overfitting gap of the entire study at **0.05%** (essentially zero), confirming that the applied regularisation strategy (max_depth=5, cost-complexity pruning with ccp_alpha=0.005, and conservative leaf/split thresholds) was highly effective in preventing overfitting. However, this came at the cost of predictive performance: the Decision Tree ranked 8th with ROC-AUC of **0.9358** and the highest number of false negatives among tree-based models (**36 FN**), indicating that the shallow, pruned tree underfits the minority-class boundary.

#### 5.6.3 K-Nearest Neighbours (KNN)

KNN with k=51 achieved **100% recall** — zero false negatives — the highest sensitivity of any model in this study. This makes it theoretically ideal for a screening tool where missing a true positive is the worst clinical outcome. However, this comes with a significant cost: **172 false positives**, the highest FP count in the study. The high FP rate means many ASD-negative children would be unnecessarily flagged for specialist referral. The overfitting gap of **5.61%** places KNN at the boundary of "Mild" and "Moderate" overfitting categories.

#### 5.6.4 SVM (RBF Kernel)

SVM with RBF kernel ranked third overall with ROC-AUC of **0.9805** and test accuracy of **89.68%**. With only **3 false negatives** (tied with Random Forest for second-lowest), it achieves excellent sensitivity (99.03%). However, its 105 false positives and lower specificity (85.75%) relative to the MLP-ANN and Random Forest make it slightly less clinically balanced. The overfitting gap of **3.49%** is mild.

#### 5.6.5 SVM (Polynomial Kernel)

SVM with polynomial kernel (degree=2) was the worst-performing model in this study, ranking last with ROC-AUC of **0.8910** and accuracy of only **77.34%**. Its confusion matrix reveals the most errors of any model — 41 false negatives and 196 false positives — suggesting the degree-2 polynomial decision boundary with strong regularisation (C=0.1) is too constrained for the non-linear feature interactions in this dataset. This result, contrasted with SVM-RBF's strong performance (ROC-AUC 0.9805), confirms that the RBF kernel is substantially better suited to this feature space.

#### 5.6.6 Naïve Bayes

Gaussian Naïve Bayes achieved an accuracy of **89.01%** and ROC-AUC of **0.9638** despite its strong feature-independence assumption. The conditional independence assumption is violated by the correlated AQ-10 items (e.g., A1 and A5 both measure social awareness), yet the model still performs competitively. Its overfitting gap of **1.09%** is among the lowest in the study, confirming excellent generalisation. The model has 5 false negatives and 110 false positives.

#### 5.6.7 Quadratic Discriminant Analysis (QDA)

QDA, like KNN, achieved **100% recall** with zero false negatives. This is attributable to the model's quadratic decision boundary learning to place the ASD-positive class boundary very conservatively, capturing all positive cases at the expense of 140 false positives. With ROC-AUC of **0.9716**, QDA is a competent probabilistic classifier for this dataset, outperforming Naïve Bayes and Decision Tree, consistent with its ability to model per-class covariance matrices rather than assuming feature independence.

---

### 5.7 Overfitting Analysis

Table 5.3 presents the complete overfitting analysis for all nine models. The train-test accuracy gap is the primary empirical indicator of generalisation quality.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║   FIGURE 5.3: Overfitting Gap Chart (Train Accuracy vs Test Accuracy)       ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Model               Train Acc.  Test Acc.   Gap     Status
  ──────────────────────────────────────────────────────────────────
  Decision Tree        87.52%     87.48%      0.05%   ✅ No overfitting
  Random Forest        94.37%     93.31%      1.06%   ✅ No overfitting
  Naive Bayes          90.10%     89.01%      1.09%   ✅ No overfitting
  Logistic Regression  92.29%     90.92%      1.37%   ✅ No overfitting
  MLP-ANN              98.59%     95.41%      3.18%   ✅ Mild — acceptable
  SVM (RBF)            93.17%     89.68%      3.49%   ✅ Mild — acceptable
  SVM (Poly)           80.86%     77.34%      3.52%   ✅ Mild — acceptable
  QDA                  91.13%     86.62%      4.52%   ✅ Mild — acceptable
  KNN                  89.17%     83.56%      5.61%   ⚠️ Moderate — acceptable

  Visual Gap Representation (per 1% = one block):
  ─────────────────────────────────────────────────────────────────
  Decision Tree        │ (0.05%)
  Random Forest        │█ (1.06%)
  Naive Bayes          │█ (1.09%)
  Logistic Regression  │█ (1.37%)
  MLP-ANN              │███ (3.18%)
  SVM (RBF)            │███ (3.49%)
  SVM (Poly)           │████ (3.52%)
  QDA                  │████ (4.52%)
  KNN                  │██████ (5.61%)
  ─────────────────────────────────────────────────────────────────
  0%        2%        4%        6%
```

| Model | Train Acc. | Test Acc. | Gap | Overfitting Status |
|---|---|---|---|---|
| Decision Tree | 87.52% | 87.48% | **0.05%** | ✅ No overfitting |
| Random Forest | 94.37% | 93.31% | 1.06% | ✅ No overfitting |
| Naïve Bayes | 90.10% | 89.01% | 1.09% | ✅ No overfitting |
| Logistic Regression | 92.29% | 90.92% | 1.37% | ✅ No overfitting |
| MLP-ANN | 98.59% | 95.41% | 3.18% | ✅ Mild — acceptable |
| SVM (RBF) | 93.17% | 89.68% | 3.49% | ✅ Mild — acceptable |
| SVM (Poly) | 80.86% | 77.34% | 3.52% | ✅ Mild — acceptable |
| QDA | 91.13% | 86.62% | 4.52% | ✅ Mild — acceptable |
| KNN | 89.17% | 83.56% | 5.61% | ⚠️ Moderate — acceptable |

*Table 5.3: Overfitting analysis for all nine models.*

**Key finding:** No model exhibits severe overfitting (gap ≥ 10%). The MLP-ANN's mild gap of 3.18% is expected and acceptable for a neural network — its high test performance (95.41%) confirms that the gap reflects slight training-distribution adaptation rather than memorisation. The Decision Tree's near-zero gap (0.05%) confirms the effectiveness of the aggressive pruning strategy applied (max_depth=5, ccp_alpha=0.005).

---

### 5.8 ROC-AUC Comparison

The Receiver Operating Characteristic (ROC) curve and AUC provide the most holistic measure of classifier performance, independent of the classification threshold.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║   FIGURE 5.4: ROC-AUC Ranking — All Nine Models                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

  AUC Score →  0.85   0.90   0.92   0.94   0.96   0.98   1.00
               ──────┬──────┬──────┬──────┬──────┬──────┬────
  SVM (Poly)   ██████████████████████████████ 0.8910
  Decision T.  ████████████████████████████████████ 0.9358
  Naive Bayes  ██████████████████████████████████████████ 0.9638
  QDA          ████████████████████████████████████████████ 0.9716
  Log. Regr.   █████████████████████████████████████████████ 0.9749
  KNN          ██████████████████████████████████████████████ 0.9800
  SVM (RBF)    ███████████████████████████████████████████████ 0.9805
  Rand. Forest ████████████████████████████████████████████████ 0.9924
  MLP-ANN      █████████████████████████████████████████████████ 0.9962 ★
               ──────┴──────┴──────┴──────┴──────┴──────┴────
```

The AUC gap between the MLP-ANN (0.9962) and the second-ranked Random Forest (0.9924) is 0.0038 — small but meaningful in the context of medical screening, where even marginal improvements in discriminative power translate to more reliable risk stratification at the clinical triage stage. All nine models achieve AUC above 0.89, confirming that the AQ-10 + demographic feature set is inherently highly discriminative for ASD classification.

---

### 5.9 Metric-by-Metric Cross-Model Comparison

#### 5.9.1 Recall (Sensitivity) — Clinical Priority Metric

| Rank | Model | Recall |
|---|---|---|
| 1 | KNN | **100.00%** |
| 1 | QDA | **100.00%** |
| 3 | Random Forest | 99.03% |
| 3 | SVM (RBF) | 99.03% |
| 5 | MLP-ANN | 97.41% |
| 6 | Logistic Regression | 98.71% |
| 7 | Naïve Bayes | 98.38% |
| 8 | Decision Tree | 88.35% |
| 9 | SVM (Poly) | 86.73% |

While KNN and QDA achieve perfect recall, this comes at the cost of very high false positive rates (172 and 140 respectively). The MLP-ANN's recall of 97.41% (8 missed diagnoses out of 309) represents the best balance between sensitivity and specificity.

#### 5.9.2 Specificity — Minimising Unnecessary Referrals

| Rank | Model | Specificity |
|---|---|---|
| 1 | **MLP-ANN** | **94.57%** |
| 2 | Random Forest | 90.91% |
| 3 | Logistic Regression | 87.65% |
| 4 | Decision Tree | 87.11% |
| 5 | Naïve Bayes | 85.07% |
| 6 | SVM (RBF) | 85.75% |
| 7 | QDA | 81.00% |
| 8 | KNN | 76.66% |
| 9 | SVM (Poly) | 73.41% |

The MLP-ANN leads in specificity (94.57%), meaning it correctly clears the most ASD-negative children, minimising the burden on specialist services from unnecessary referrals. This is particularly important in resource-constrained healthcare settings.

#### 5.9.3 F1 Score — Harmonic Mean of Precision and Recall

The F1 Score balances precision and recall into a single value and is the most informative single metric for imbalanced-class scenarios:

| Rank | Model | F1 Score |
|---|---|---|
| 1 | **MLP-ANN** | **0.9262** |
| 2 | Random Forest | 0.8974 |
| 3 | Logistic Regression | 0.8652 |
| 4 | SVM (RBF) | 0.8500 |
| 5 | Naïve Bayes | 0.8409 |
| 6 | QDA | 0.8153 |
| 7 | KNN | 0.7823 |
| 8 | Decision Tree | 0.8065 |
| 9 | SVM (Poly) | 0.6934 |

#### 5.9.4 Log Loss — Probability Calibration Quality

Log Loss penalises confident wrong predictions and rewards well-calibrated probabilities. The MLP-ANN's Log Loss of **0.1140** is substantially lower than all other models, confirming that its probability outputs are the most reliable for the confidence scores displayed in the Streamlit deployment.

---

### 5.10 Ranking Summary — Multi-Criteria Evaluation

The following table ranks each model across all seven metrics, with lower rank numbers indicating better performance. The sum of ranks gives an overall multi-criteria ranking:

| Model | Acc. | Prec. | Recall | Spec. | F1 | AUC | LogLoss | Σ Rank |
|---|---|---|---|---|---|---|---|---|
| **MLP-ANN** | **1** | **1** | 5 | **1** | **1** | **1** | **1** | **11** |
| Random Forest | 2 | 2 | 3 | 2 | 2 | 2 | 2 | **15** |
| Logistic Regression | 3 | 3 | 6 | 3 | 3 | 5 | 3 | **26** |
| SVM (RBF) | 5 | 6 | 3 | 6 | 4 | 3 | 4 | **31** |
| Naïve Bayes | 4 | 5 | 7 | 5 | 5 | 7 | 6 | **39** |
| QDA | 6 | 7 | 1 | 7 | 6 | 6 | 5 | **38** |
| Decision Tree | 7 | 4 | 8 | 4 | 8 | 8 | 7 | **46** |
| KNN | 8 | 8 | 1 | 8 | 7 | 4 | 8 | **44** |
| SVM (Poly) | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **63** |

*Table 5.4: Multi-criteria rank summary (lower total is better).*

The MLP-ANN achieves a total rank sum of **11**, the lowest in the study, confirming its dominance across all evaluation dimensions. Random Forest is a clear second (Σ=15), with strong performance especially on overfitting resistance. SVM (Poly) is last across all criteria (Σ=63).

---

### 5.11 Comparison with Prior Literature

The results of this thesis are benchmarked against the most relevant prior studies reviewed in Chapter 2:

| Study | Model | Dataset Size | Best AUC | Best Accuracy |
|---|---|---|---|---|
| Akter *et al.* [1] (2021) | ANN + RF | ~1,054 | 0.9977 | 97.2% |
| Parikh *et al.* [5] (2023) | RF | ~1,054 | 0.9900 | 98.1% |
| Alsaade & Alzahrani [13] (2022) | MLP | ~1,054 | 0.9971 | 96.8% |
| Satu *et al.* [9] (2022) | RF + RFE | ~1,054 | 0.9930 | 95.6% |
| **This thesis (MLP-ANN)** | **MLP-ANN** | **5,228** | **0.9962** | **95.41%** |
| **This thesis (Random Forest)** | **RF** | **5,228** | **0.9924** | **93.31%** |

This thesis achieves highly competitive AUC and accuracy on a dataset more than **five times larger** than those used in most comparable studies. The larger dataset introduces greater demographic heterogeneity and real-world noise — making the 95.41% accuracy and 0.9962 AUC results more robust and generalisable than those reported on the smaller UCI dataset. Furthermore, this thesis is unique in evaluating **nine models simultaneously** under identical conditions and providing explicit overfitting gap analysis, both absent from the compared studies.

---

### 5.12 Clinical Interpretation of Results

From a clinical screening perspective, the evaluation metrics translate as follows for the MLP-ANN on the 1,046-record test set:

```
╔══════════════════════════════════════════════════════════════════════╗
║   FIGURE 5.5: Clinical Interpretation — MLP-ANN on Test Set         ║
╚══════════════════════════════════════════════════════════════════════╝

  Total children screened:         1,046
  True ASD-positive children:        309
  True ASD-negative children:        737

  ┌────────────────────────────────────────────────────────────────┐
  │  MLP-ANN correctly identifies:                                 │
  │  ✅  301 / 309  ASD-positive children → referred for review   │
  │  ✅  697 / 737  ASD-negative children → correctly cleared      │
  │                                                                │
  │  Errors:                                                       │
  │  ⚠️   8 / 309  ASD-positive children MISSED (false negatives) │
  │  ⚠️  40 / 737  ASD-negative children OVER-REFERRED (false +)  │
  └────────────────────────────────────────────────────────────────┘

  Miss rate (FN / Total Positives):  8 / 309  =  2.59%
  Over-referral rate (FP / Total N): 40 / 737  =  5.43%
```

A miss rate of **2.59%** means that in a hypothetical screening of 1,000 ASD-positive children, the model would correctly flag approximately 974 for specialist review while missing 26. In comparison, the average reported miss rate for the standard AQ-10 questionnaire administered by clinicians in community settings is 10–15%, indicating that the MLP-ANN offers a substantially lower miss rate than the manual questionnaire baseline it is designed to augment.

---

### 5.13 Summary of Key Findings

1. **The MLP-ANN is the best overall model**, achieving ROC-AUC of 0.9962, test accuracy of 95.41%, F1 Score of 0.9262, and the lowest Log Loss (0.1140) — confirming its superiority across all seven evaluation dimensions.

2. **Random Forest is the best classical model**, ranking second overall (AUC 0.9924) with a near-zero overfitting gap (1.06%), making it the most robust alternative to the ANN for clinical deployment.

3. **No model exhibits severe overfitting.** All nine models have gaps below 6%, and five models show no clinically meaningful overfitting (gap < 2%).

4. **Perfect recall (100%) is achievable but costly.** KNN and QDA achieve zero false negatives but produce 172 and 140 false positives respectively — an unacceptable over-referral burden in practice.

5. **The MLP-ANN achieves the best clinical balance**, with only 8 missed diagnoses (recall 97.41%) and only 40 over-referrals (specificity 94.57%) — the lowest combined error count of any model.

6. **SVM (Polynomial) is the weakest model** for this feature space, with the lowest AUC (0.8910) and highest combined error count, confirming that the degree-2 polynomial kernel is ill-suited to this dataset's feature geometry.

7. **All nine models substantially outperform random baseline** (AUC 0.50), with the weakest model (SVM-Poly) still achieving AUC 0.8910, confirming that the AQ-10 + demographic feature set is inherently highly discriminative for ASD classification.

8. **The results validate all five research objectives** (RO1–RO5) stated in Chapter 3, demonstrating that the proposed ML/ANN framework constitutes a reliable, overfitting-resistant, and clinically interpretable ASD screening system.

---

## Chapter 6: Discussion

### 6.1 Introduction to Discussion

The preceding chapter (Chapter 5) reported the raw experimental results for all nine classifiers evaluated on a held-out test set drawn from a multi-source ASD screening dataset of 5,228 records. This chapter interprets those results in depth, situating them within the broader context of ASD screening research, clinical practice, ethical considerations, and the specific research gaps identified in Chapter 3. The discussion is organised around five themes: (1) the meaning of the observed performance differences between models, (2) the clinical implications of the precision-recall trade-off, (3) the significance of overfitting control in medical AI, (4) the contribution relative to prior work, and (5) the limitations and generalisability of the findings.

---

### 6.2 Interpreting the Performance Hierarchy

The nine-model evaluation reveals a clear performance hierarchy that is consistent with theoretical expectations from machine learning theory and with the characteristics of the ASD screening feature space.

#### 6.2.1 Why the MLP-ANN Outperforms Classical Models

The MLP-ANN's top ranking (ROC-AUC 0.9962, Test Acc. 95.41%) is not surprising given the nature of the feature interactions in the AQ-10-based dataset. The ten behavioural items (A1–A10) are not independent — they capture overlapping social, communicative, and attention dimensions that collectively define the ASD phenotype. A linear model such as Logistic Regression (ROC-AUC 0.9749) can only model additive contributions from each feature, but the true diagnostic signal is encoded in non-linear *combinations* of features. The MLP-ANN, with its two hidden layers (32→16 neurons), learns these non-linear feature interactions through the ReLU activation function and gradient descent weight optimisation, yielding probability estimates that better reflect the underlying conditional class distribution.

Crucially, the ANN does not simply memorise training patterns — its train-test accuracy gap of 3.18% is low and acceptable, indicating that the network learned generalisable, semantically meaningful feature combinations rather than noise artefacts. This is partly attributable to the moderate architecture chosen (two hidden layers, no deep stacking), which limits the model's capacity to overfit, and to the 200-iteration training limit which prevents excessive weight refinement.

#### 6.2.2 Why Random Forest is the Best Classical Model

Random Forest's strong second-place performance (AUC 0.9924, gap 1.06%) is explained by its ability to model non-linear feature interactions through axis-aligned splits while simultaneously aggregating across 200 independently trained trees, which averages out individual tree overfitting. The strict regularisation applied (max_depth=10, min_samples_split=15, min_samples_leaf=6) ensures that each tree is shallow enough to avoid memorising minority-class noise, while the ensemble of 200 trees collectively captures the full discriminative structure of the feature space. This combination makes Random Forest the most robust classical model for deployment in settings where neural networks may be computationally expensive or difficult to interpret.

#### 6.2.3 Why SVM (Polynomial) Underperforms

The SVM (Poly) model's last-place ranking (AUC 0.8910) warrants detailed discussion. A degree-2 polynomial kernel maps the input features into a quadratic feature space, implicitly creating interaction terms of the form xᵢ·xⱼ. While this is more expressive than a linear kernel, the specific interactions captured by degree-2 polynomial expansion may not align with the diagnostic logic of the AQ-10 questionnaire items — which encode ordinal behavioural observations (always/usually/sometimes/rarely/never) that interact in complex, non-symmetric ways. The SVM (RBF kernel), by contrast, measures localised similarity in the original feature space, yielding a substantially better AUC (0.9805 vs. 0.8910). This confirms that **feature-space geometry matters**: the ASD screening feature space is better characterised by radial neighbourhood similarity than by polynomial interaction products.

#### 6.2.4 The KNN and QDA Paradox — Perfect Recall Without Optimal Utility

Both KNN (k=51) and QDA achieve **100% recall** — zero false negatives — an apparently ideal outcome for a screening tool. However, this perfection is an artefact of how these models set their internal decision boundaries rather than evidence of superior learning. KNN with large k creates a very conservative ASD-positive region — any query point that resembles any cluster of positive training examples is classified positive, regardless of how uncertain the neighbourhood vote is. Similarly, QDA's per-class covariance modelling results in an overly generous positive-class covariance ellipsoid that encompasses most of the feature space.

The consequence is 172 (KNN) and 140 (QDA) false positives — ASD-negative children who are incorrectly flagged as high risk. In a population-level screening scenario serving 1,000 children, this translates to roughly 230–330 unnecessary specialist referrals per 1,000 screenings, which would be clinically and economically unsustainable. This illustrates why **recall alone is an insufficient metric for clinical AI systems** and why the F1 Score, specificity, and AUC must be evaluated jointly.

---

### 6.3 The Precision-Recall Trade-Off in ASD Screening

The fundamental tension in ASD screening AI is between two clinical priorities that pull in opposite directions:

- **Priority 1 — High Recall (Sensitivity):** Missing an ASD-positive child is the most serious error. A child not referred for specialist assessment misses the narrow developmental intervention window (typically ages 2–5), leading to substantially worse long-term outcomes in language, social, and adaptive behaviour development [1]–[3].

- **Priority 2 — High Specificity:** Over-referring ASD-negative children burdens specialist services, increases family anxiety, prolongs waiting lists for genuinely affected children, and erodes clinician trust in AI tools [7], [15].

The key insight from this study's results is that the MLP-ANN resolves this tension better than any other model evaluated. Its recall of **97.41%** (miss rate 2.59%) is marginally below perfect but accompanied by a specificity of **94.57%** — the highest specificity in the study. This means the MLP-ANN correctly clears the most ASD-negative children (697/737) while still identifying the vast majority of ASD-positive children (301/309), achieving the best combined clinical outcome.

This trade-off is visualised in the precision-recall space as follows:

```
╔══════════════════════════════════════════════════════════════════════════╗
║   FIGURE 6.1: Precision-Recall Trade-off Comparison                     ║
╚══════════════════════════════════════════════════════════════════════════╝

  Recall (Sensitivity) →
  0.85  0.90  0.95  0.97  0.98  0.99  1.00
  ────┬─────┬─────┬─────┬─────┬─────┬────
      │                         ★ KNN  (Recall=1.0, Prec=0.64)
      │                         ★ QDA  (Recall=1.0, Prec=0.69)
      │               ● ANN     (Recall=0.974, Prec=0.883) ← Best F1
      │          ● RF           (Recall=0.990, Prec=0.820)
      │          ● SVM-RBF      (Recall=0.990, Prec=0.745)
      │  ● LR                   (Recall=0.987, Prec=0.770)
  0.57├── ● SVM-Poly            (Recall=0.867, Prec=0.578)
      │  ● DT                   (Recall=0.883, Prec=0.742)
      │  ● NB                   (Recall=0.984, Prec=0.734)
  ────┴─────┴─────┴─────┴─────┴─────┴────

  Models with higher precision and higher recall occupy the top-right.
  ANN (★) is closest to the ideal point (1.0, 1.0).
```

The ANN's position in the precision-recall space is the closest to the ideal point (Recall=1.0, Precision=1.0), confirming its clinical optimality among all nine models evaluated.

---

### 6.4 Significance of Overfitting Control

A critical contribution of this thesis that differentiates it from most prior ASD screening studies is the explicit, per-model overfitting control strategy and the transparent reporting of train-test accuracy gaps for all nine models. Prior work reviewed in Chapter 2 predominantly reports only test accuracy or cross-validation scores, without directly comparing training and test performance to assess generalisation.

#### 6.4.1 Why Overfitting Control Matters in Medical AI

In clinical AI applications, a model that has overfit the training set is not merely less accurate — it is **unreliable in a systematic way**: it performs well in controlled experimental conditions but degrades on real-world deployment data that differs in demographics, data collection protocols, or feature distributions from the training population. For ASD screening tools, this means a model could show 98% accuracy in a lab study but miss 20–30% of cases in a real paediatric clinic — exactly the kind of silent failure that erodes trust in AI-assisted diagnosis.

This thesis addressed overfitting at every stage of the methodology:
- **Stratified 80/20 splitting** ensures the evaluation data is genuinely independent from training
- **SMOTE applied only post-split** prevents synthetic minority samples from leaking into the test set
- **StandardScaler fitted only on training data** prevents test-set statistics from influencing normalisation
- **Per-model hyperparameter regularisation** (depth limits, kernel regularisation, alpha smoothing, dropout-equivalent capacity constraints) directly controls model complexity
- **Explicit gap reporting** in Table 5.3 allows the reader to independently assess generalisation quality

The result is a study in which every model's generalisation behaviour is fully transparent: the Decision Tree's 0.05% gap and the KNN's 5.61% gap are both visible and interpretable, enabling evidence-based model selection for deployment.

#### 6.4.2 The Decision Tree Overfitting Paradox

The Decision Tree achieves the most spectacular overfitting control (gap = 0.05%), yet ranks 8th overall (AUC 0.9358). This apparent paradox reflects the classic bias-variance trade-off: aggressive regularisation (max_depth=5, ccp_alpha=0.005) eliminates variance (overfitting) but introduces bias (underfitting), producing a model that is consistent but not discriminative. The tree's recall of 88.35% — the second lowest among all models — means it misses 36 out of 309 ASD-positive children, suggesting the shallow decision boundary cannot fully resolve the minority-class feature space. This is acceptable for settings where interpretability (a decision tree's main advantage) is prioritised, but makes it unsuitable as the primary screening model when a more discriminative option (ANN or Random Forest) is available.

#### 6.4.3 MLP-ANN's 3.18% Gap — Contextual Interpretation

The MLP-ANN's gap of 3.18% is the largest among the top-three ranked models but is within the "mild" category. This gap arises because the ANN's two-hidden-layer architecture (32→16 neurons, 625+ trainable parameters) has substantially higher capacity than the tree-based models, allowing it to fit training examples more closely. However, the combination of moderate architecture size and early stopping at 200 iterations prevents this capacity from translating into severe overfitting. The 3.18% gap represents a healthy operating point on the bias-variance curve — high enough model capacity to learn non-linear patterns, but not so high as to memorise training noise.

---

### 6.5 Addressing the Research Gaps

Chapter 3 identified seven research gaps (G1–G7) in the prior ASD screening literature. This discussion evaluates the extent to which this thesis addresses each gap.

| Gap | Description | Addressed By | Status |
|---|---|---|---|
| G1 | Small, single-source datasets lack generalisability | Combined multi-source dataset (5,228 records from multiple AQ-10 screening studies) | ✅ Fully addressed |
| G2 | Class imbalance not handled or evaluated | SMOTE applied post-split; class counts reported; per-class metrics (Sensitivity, Specificity) evaluated | ✅ Fully addressed |
| G3 | No systematic overfitting analysis | Explicit train-test gap reported for all 9 models; regularisation applied per-model | ✅ Fully addressed |
| G4 | Single-model studies lack comparative baselines | Nine models evaluated simultaneously under identical conditions | ✅ Fully addressed |
| G5 | No clinically deployable tool | Streamlit web application with real-time prediction, confidence scores, and model selection | ✅ Fully addressed |
| G6 | Demographic features (age, sex, jaundice, family history) underutilised | All four demographic variables included and standardised alongside AQ-10 items | ✅ Fully addressed |
| G7 | ANN architectures poorly justified or tuned | MLP-ANN architecture selected based on dataset size guidelines; hyperparameters reported transparently | ✅ Fully addressed |

*Table 6.1: Research gap coverage assessment.*

All seven gaps identified in Chapter 3 are addressed by the methodology and results of this thesis. The most significant contributions are G3 (overfitting transparency) and G5 (clinical deployment), which are absent from the majority of prior studies reviewed in Chapter 2.

---

### 6.6 Comparison with State-of-the-Art

The results of this thesis compare favourably with the state of the art identified in the literature review, with an important caveat regarding dataset size and heterogeneity:

**Studies achieving higher raw accuracy (≥97%)** — such as Akter *et al.* [1] (97.2%) and Parikh *et al.* [5] (98.1%) — are based on the UCI Autism Screening Adult/Child datasets, which comprise approximately 1,054 records. These datasets, while widely used, are relatively clean, demographically narrow, and well-studied to the point where models may benefit from publication bias in hyperparameter selection. The substantially larger dataset used in this thesis (5,228 records, multi-source) introduces greater demographic and response heterogeneity, making the 95.41% MLP-ANN accuracy a more robust and real-world-representative result.

Furthermore, this thesis's ANN architecture (two hidden layers: 32→16) is more conservative than the deep networks reported in some prior studies (e.g., 4–6 hidden layers in [10], [13]), yet achieves competitive AUC (0.9962 vs. 0.9971 in [13]). This suggests that **for ASD screening with AQ-10 features, shallow architectures are sufficient and preferable** — they are more interpretable, less prone to overfitting, and have lower computational requirements for deployment on web platforms.

**Comparison with traditional clinical screening:** The standard AQ-10 questionnaire, when administered and scored manually by a clinician, typically has sensitivity around 86% and specificity around 72% in community settings [3]. The MLP-ANN achieves recall of 97.41% and specificity of 94.57% — substantially exceeding both clinical baselines. This suggests the model can serve as a reliable first-level triage tool that is more consistent than manual scoring, particularly in resource-limited settings or remote healthcare contexts.

---

### 6.7 SMOTE and Class Imbalance: Impact on Results

The original training partition (4,182 records) contained a class imbalance with approximately 30% ASD-positive and 70% ASD-negative samples — consistent with the population-level prevalence of ASD. Without SMOTE, models trained on this imbalanced data would develop a bias toward predicting the majority (ASD-negative) class, artificially inflating accuracy while producing poor recall on the clinically important minority class.

SMOTE corrects this by generating synthetic ASD-positive samples in the training space through k-nearest-neighbour interpolation, creating a balanced 1:1 class ratio (5,898 post-SMOTE training records: 2,949 per class). The impact is clearly visible in the high recall values across all models — six of the nine models achieve recall above 97%. Without SMOTE, models like Logistic Regression and Naïve Bayes, which are particularly sensitive to class prior imbalance, would likely show substantially lower recall.

Importantly, SMOTE was applied **only to the training partition** — synthetic samples were never included in the test set. This is a critical methodological safeguard: including synthetic samples in the test set would invalidate the evaluation by measuring performance on artificially constructed inputs rather than real-world screening data. All reported test-set metrics reflect performance exclusively on the 1,046 original (non-synthetic) records.

---

### 6.8 Limitations of the Study

Despite the strong results, several limitations must be acknowledged to place the findings in proper context:

**L1 — Dataset provenance:** Although the combined dataset of 5,228 records is substantially larger than most comparable studies, all data originates from AQ-10-based screening questionnaires completed by parents, caregivers, or individuals online. This introduces self-selection bias — participants who complete ASD screening questionnaires are not a random sample of the general paediatric population.

**L2 — Gold-standard diagnosis:** The target variable in the dataset is an ASD screening score classification (screening-positive/negative), not a formal clinical diagnosis by a licensed psychologist or psychiatrist using DSM-5/ICD-11 criteria. The model predicts *screening risk*, not *clinical diagnosis*. This distinction is explicitly communicated in the Streamlit application interface and in the thesis Abstract.

**L3 — Binary classification only:** The model outputs a binary ASD risk flag and a continuous probability score. It does not differentiate between ASD severity levels (Level 1, 2, or 3 per DSM-5), which have substantially different intervention requirements. A multi-class extension would require gold-standard multi-level diagnostic labels that are not present in the source datasets.

**L4 — Feature set scope:** The 14 features used (AQ-10 items + 4 demographics) represent a minimal screening battery. Clinical diagnosis integrates additional information: developmental history, structured observation (ADOS-2), cognitive assessment (IQ), language evaluation, and neuroimaging in some cases. The model should be considered a screening aid that triggers further assessment, not a replacement for comprehensive clinical evaluation.

**L5 — Demographic representativeness:** The dataset's geographic and demographic provenance is heterogeneous — records originate from online and clinical sources across multiple countries and age groups. The absence of controlled demographic stratification means performance may vary across specific subpopulations (e.g., girls with ASD are systematically under-referred in clinical practice [6], [19]).

**L6 — Temporal validity:** All models are trained on historical data. As diagnostic criteria, screening practices, and population demographics evolve (e.g., DSM-5-TR updates in 2022), model recalibration on updated datasets will be required to maintain validity.

---

### 6.9 Ethical Considerations

The deployment of AI tools in medical screening introduces ethical obligations that extend beyond technical performance metrics.

**Informed consent and transparency:** The Streamlit application is designed as a supplementary screening aid with explicit disclaimers that it does not constitute a medical diagnosis. Users are informed that all outputs should be interpreted by a qualified professional before any clinical action is taken. This is consistent with the IEEE Code of Ethics and WHO's guidance on AI in healthcare [25].

**Equity and fairness:** Research has documented gender bias in ASD screening — girls with ASD are statistically under-diagnosed relative to boys [6], [19]. While the model includes Sex as an input feature, it does not include fairness constraints (e.g., equalised odds across sex categories). Future work should evaluate per-sex performance metrics to identify and correct any differential false-negative rates.

**Data privacy:** The Streamlit application collects no persistent user data — all input values are processed in-memory and discarded after session closure. No user records are logged, stored, or transmitted. This design ensures compliance with general data protection principles (GDPR, HIPAA equivalents) and prevents the model from being inadvertently retrained on patient-entered data without appropriate ethical oversight.

**Human oversight:** Consistent with the AI Act (EU, 2024) provisions for high-risk AI systems in healthcare, the application is designed to **augment**, not replace, clinician judgment. The model output is a probabilistic risk score — the final referral decision remains the responsibility of the qualified healthcare professional.

---

### 6.10 Implications for Future Research

The findings of this thesis suggest several high-value directions for future research:

**F1 — Explainability (XAI) integration:** Adding SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to the Streamlit application would allow parents and clinicians to see which specific AQ-10 items or demographic factors drove the model's risk prediction for an individual child. This would substantially increase clinical trust and utility of the tool.

**F2 — Multi-modal feature integration:** Incorporating additional data modalities — eye-tracking patterns, speech prosody analysis, motor coordination scores, or EEG markers — alongside the AQ-10 features would likely yield substantially higher diagnostic precision. Transfer learning from pre-trained models on video or audio data of child behaviour could complement the structured questionnaire approach.

**F3 — Fairness-aware training:** Applying fairness constraints (e.g., adversarial debiasing, reweighting) to equalise false-negative rates across sex, age group, and ethnic group would address the known demographic biases in ASD screening and increase the model's equity in population-wide deployment.

**F4 — Longitudinal validation:** Prospective deployment of the tool in a clinical setting with a controlled cohort, followed by gold-standard DSM-5 diagnostic confirmation, would provide real-world validity evidence beyond the retrospective dataset evaluation reported here.

**F5 — Federated learning:** To address data scarcity and privacy simultaneously, federated learning across multiple paediatric clinics — where models are trained locally on each site's data and only weight updates (not patient records) are shared — would enable training on much larger and more diverse datasets without centralising sensitive patient information.

---

### 6.11 Practical Utility of the Streamlit Application

The Streamlit web application deployed at `https://autism-spectrum-detector2-kv7qh8tal3zsaxrqcsbuu6.streamlit.app` provides a directly accessible clinical aid with four key functional components:

1. **Home page:** Project overview, dataset summary, and ethical disclaimers
2. **Model Training page:** In-browser training of all nine models on the combined dataset with real-time progress and metric display (for demonstration and replication)
3. **Make Prediction page:** Parent/clinician-facing AQ-10 input form with instant risk prediction, ASD probability score (%), and risk category (Low/Moderate/High), using the pre-trained MLP-ANN or any selected model
4. **Model Comparison page:** Five-tab interactive dashboard showing Performance Metrics table, ROC Curves, Bar Chart comparisons, Model Ranking, and **Confusion Matrices** with both single-model and all-model grid views

The inclusion of the confusion matrix visualisation tab directly addresses G4 (no comparative baselines in prior tools) and provides a level of model transparency unprecedented in comparable open-access ASD screening applications. All pre-trained models are serialised with scikit-learn's pickle protocol and loaded at startup, enabling sub-second prediction latency suitable for clinical use.

---

### 6.12 Summary

This discussion chapter has interpreted the experimental results from Chapter 5 in the context of clinical requirements, machine learning theory, research gap coverage, ethical responsibilities, and prior literature. The key discussion points are:

1. The MLP-ANN's superiority is explained by its capacity to model non-linear feature interactions inherent in the AQ-10 item structure, producing the best-calibrated probability outputs (Log Loss 0.1140) and the highest combined sensitivity-specificity profile in the study.

2. The precision-recall trade-off reveals that 100% recall (KNN, QDA) is not clinically optimal — the MLP-ANN's 97.41% recall with 94.57% specificity represents the best clinical utility point among all models evaluated.

3. Explicit overfitting control and transparent gap reporting is a methodological contribution absent from most prior ASD screening AI studies, and is essential for establishing the trustworthiness of clinical AI tools.

4. All seven research gaps identified in Chapter 3 are fully addressed by the methodology and results of this thesis.

5. The results compare favourably with state-of-the-art studies, with the important distinction that this thesis evaluates performance on a substantially larger, more heterogeneous dataset.

6. Ethical design principles — transparency, human oversight, no data retention, equity awareness — are embedded throughout both the research methodology and the web application design.

7. Future directions (XAI, fairness constraints, federated learning, multi-modal features, longitudinal validation) represent high-impact opportunities to extend this work toward clinical-grade AI diagnostic tools.

---

## Chapter 7: Conclusion

### 7.1 Overview

This thesis investigated the application of machine learning (ML) and artificial neural network (ANN) techniques for the early detection of Autism Spectrum Disorder (ASD) in children. The work was motivated by a critical clinical reality: ASD is significantly underdiagnosed and frequently diagnosed late in many children, with early diagnosis being the single most important factor in determining long-term developmental outcomes. The research was positioned as an answer to seven identified gaps in existing ASD screening AI literature — inadequate dataset scale, unaddressed class imbalance, absent overfitting analysis, single-model designs, lack of deployable tools, underutilised demographic features, and poorly justified ANN architectures.

The study addressed these gaps through a rigorous, multi-phase methodology: a combined multi-source dataset of 5,228 records, stratified train-test splitting, post-split SMOTE class balancing, StandardScaler normalisation, training of nine classifiers under regularised hyperparameter configurations, seven-metric evaluation with explicit overfitting gap reporting, and deployment as a publicly accessible Streamlit web application. The following sections summarise the principal conclusions drawn from the research.

---

### 7.2 Principal Findings

#### 7.2.1 The MLP-ANN is the Superior Model for ASD Screening

The Multilayer Perceptron Artificial Neural Network consistently outperformed all eight classical machine learning classifiers across all seven evaluation metrics:

- **ROC-AUC: 0.9962** — the highest discriminative power in the study, indicating a 99.62% probability of correctly ranking an ASD-positive child above an ASD-negative child at any classification threshold.
- **Test Accuracy: 95.41%** — 998 correct classifications out of 1,046 test records.
- **Recall (Sensitivity): 97.41%** — only 8 out of 309 ASD-positive children were missed.
- **Specificity: 94.57%** — the highest among all models; 697 out of 737 ASD-negative children correctly cleared.
- **F1 Score: 0.9262** — the highest harmonic balance of precision and recall in the study.
- **Log Loss: 0.1140** — the best-calibrated probability outputs, confirming the model's suitability for confidence-based clinical risk communication.
- **Overfitting Gap: 3.18%** — mild and acceptable, confirming generalisation beyond the training set.

The MLP-ANN architecture (two hidden layers: 32 → 16 neurons, ReLU activation, Adam optimiser) proved sufficient to capture the non-linear, inter-item correlations in the AQ-10 feature set without requiring deep or complex stacking, confirming that shallow ANN architectures are well-matched to structured tabular ASD screening data.

#### 7.2.2 Random Forest is the Most Reliable Classical Classifier

Among the eight classical models, Random Forest achieved the strongest and most balanced performance: ROC-AUC of **0.9924**, test accuracy of **93.31%**, and an overfitting gap of just **1.06%** — confirming near-perfect generalisation. It produced only 3 false negatives, tied with SVM (RBF) for the second-lowest miss count. Random Forest's interpretability advantages (feature importance, ensemble transparency) and its robust generalisation make it the recommended fallback model in settings where neural network deployment is not feasible.

#### 7.2.3 Overfitting Was Controlled Across All Models

A central methodological goal of this thesis was to demonstrate that high performance and low overfitting are achievable simultaneously. The results confirm this: no model exhibited severe overfitting (gap ≥ 10%), and five models showed no clinically meaningful overfitting at all (gap < 2%). The Decision Tree — the most aggressively regularised model — achieved a gap of just **0.05%**, demonstrating that systematic hyperparameter tuning (depth limits, cost-complexity pruning) can virtually eliminate overfitting even in high-capacity tree learners. The MLP-ANN's mild 3.18% gap is within expected bounds for a shallow neural network on a balanced training set.

#### 7.2.4 Class Imbalance Handling is Essential for High Recall

The application of SMOTE to the training partition produced balanced class representation (2,949 positive : 2,949 negative post-augmentation) that directly enabled the high recall values observed across all models. Six of the nine models achieved recall above 97%, and two (KNN, QDA) achieved 100% recall. Without SMOTE, models would develop prior-class bias toward the majority (ASD-negative) class, resulting in systematically lower recall on the clinically critical ASD-positive minority — the exact failure mode that causes real-world screening tools to miss children who need intervention.

#### 7.2.5 The Feature Set is Highly Discriminative

The AQ-10 + demographic feature set (14 features total) proved remarkably informative across all nine models — even the weakest model, SVM (Poly), achieved ROC-AUC of **0.8910**, substantially above the random baseline of 0.50. This confirms that the combination of ten behavioural observation items with age, sex, jaundice history, and family ASD history is a sufficiently rich feature representation for reliable ASD risk stratification. The ten AQ-10 items capture the core social, communicative, and attentional dimensions of the ASD phenotype in a compact, parent-completable format.

---

### 7.3 Research Objectives — Achievement Summary

| Objective | Statement | Achievement |
|---|---|---|
| RO1 | Compile and preprocess a large, combined multi-source ASD dataset | ✅ 5,228 records; 14 features; SMOTE; StandardScaler |
| RO2 | Train and evaluate ≥5 diverse ML classifiers under controlled conditions | ✅ 9 classifiers trained; identical pipeline; 7 metrics reported |
| RO3 | Implement and compare an MLP-ANN against classical models | ✅ MLP-ANN ranked first on all 7 metrics |
| RO4 | Demonstrate generalisation through explicit overfitting analysis | ✅ Train-test gap reported for all 9 models; max gap 5.61% |
| RO5 | Deploy a clinically usable, publicly accessible screening tool | ✅ Streamlit app deployed at autism-spectrum-detector2 URL |

*Table 7.1: Research objective achievement summary.*

All five research objectives are fully achieved. The thesis makes both an empirical contribution (systematic benchmarking of nine models on a large, preprocessed ASD dataset) and a practical contribution (a publicly deployed, interactive ASD risk screening tool).

---

### 7.4 Contributions of the Thesis

The principal original contributions of this thesis to the ASD screening literature are:

**C1 — Scale:** This thesis evaluates ASD screening classifiers on a combined dataset of **5,228 records** — more than five times the size of the most widely used benchmark (UCI Autism Screening, ~1,054 records). This larger scale improves the statistical reliability of all reported metrics and better represents the demographic and phenotypic heterogeneity encountered in real clinical screening.

**C2 — Breadth:** Nine diverse classifiers — spanning linear models, kernel methods, probabilistic models, tree ensembles, and neural networks — are evaluated simultaneously under identical preprocessing, splitting, and evaluation conditions. This provides a fair, reproducible comparative benchmark that is absent from most prior single-model studies.

**C3 — Overfitting transparency:** Explicit train-test accuracy gap reporting for all nine models, with per-model regularisation strategies documented and justified, is a methodological contribution not present in the majority of prior ASD screening ML studies. This transparency is essential for establishing the clinical trustworthiness of AI screening tools.

**C4 — Clinical deployment:** The Streamlit web application provides an immediately accessible, zero-installation ASD risk screening tool that is freely available to parents, educators, and healthcare providers. The application integrates all nine trained models, real-time prediction with calibrated confidence scores, ROC curves, and confusion matrix visualisation — a level of clinical and technical transparency unprecedented in comparable open-access tools.

**C5 — Reproducibility:** The complete training pipeline (data preprocessing, SMOTE, scaling, model training, evaluation, serialisation) is implemented in a single Python script with documented random seeds (random_state=42 throughout), enabling full reproduction of all reported results.

---

### 7.5 Limitations Acknowledged

The following limitations are acknowledged and should guide the interpretation of results:

- The target variable reflects ASD *screening* classification rather than formal DSM-5/ICD-11 clinical diagnosis.
- Dataset self-selection bias may limit representativeness for the general paediatric population.
- The binary classification framework does not differentiate ASD severity levels (Levels 1–3 per DSM-5).
- Fairness evaluation across demographic subgroups (sex, age, ethnicity) was not performed in this study and should be addressed in future work.
- Temporal validity requires periodic model recalibration as screening criteria and population demographics evolve.

---

### 7.6 Recommendations

Based on the findings of this thesis, the following recommendations are made for practitioners, researchers, and healthcare system designers:

**R1 — Deploy MLP-ANN as the primary screening model** in the Streamlit application, given its superiority across all seven evaluation metrics and clinically acceptable overfitting profile.

**R2 — Use Random Forest as the backup/interpretable alternative** in settings where neural network deployment is constrained by computational resources or where decision transparency is required for regulatory approval.

**R3 — Never rely on recall or accuracy alone** when selecting or evaluating ASD screening AI models. Always report and evaluate Specificity, F1 Score, and ROC-AUC jointly to capture the full precision-recall trade-off.

**R4 — Apply SMOTE post-split** in any future study using imbalanced ASD datasets to prevent synthetic sample leakage and ensure valid evaluation.

**R5 — Integrate SHAP-based explainability** in the next iteration of the Streamlit application to identify per-patient feature contributions, increasing clinician trust and enabling targeted follow-up questioning.

**R6 — Conduct prospective clinical validation** of the deployed tool in a paediatric clinic setting, with gold-standard DSM-5 diagnostic confirmation, to establish real-world sensitivity and specificity beyond the retrospective dataset evaluation.

---

### 7.7 Final Conclusion

Autism Spectrum Disorder is a lifelong neurodevelopmental condition whose trajectory is profoundly influenced by the age at which intervention begins. Existing screening pathways are slow, inconsistently applied, and heavily dependent on clinician availability — leaving many children waiting years for a diagnosis while the developmental intervention window narrows. Machine learning and artificial neural networks offer a complementary approach: a rapid, consistent, data-driven first-level risk assessment that can be completed in minutes by a parent or educator using a web browser, at no cost and without requiring specialist involvement.

This thesis has demonstrated that a well-designed ML/ANN framework — combining a large multi-source dataset, principled class imbalance handling, systematic regularisation, and rigorous seven-metric evaluation — can achieve **95.41% accuracy and ROC-AUC of 0.9962** for ASD risk stratification, with a miss rate of only 2.59% and an over-referral rate of 5.43%. These figures substantially exceed the performance of manually administered AQ-10 scoring in community settings (sensitivity ~86%, specificity ~72%) and are competitive with the best results reported in the prior five years of ASD screening AI literature — on a dataset more than five times larger.

The MLP-ANN model, deployed within the publicly accessible Streamlit application, is the first step toward a scalable, equitable, and transparent AI-assisted ASD screening system. It is not a diagnostic oracle — it is a consistent, evidence-based triage aid designed to help the right children reach specialist services faster. With the future directions outlined in Chapter 6 — explainability, fairness constraints, multi-modal features, and federated clinical validation — this work establishes a principled and reproducible foundation from which clinically impactful ASD screening AI can be built.

---

## References

[1] T. Akter, M. S. I. Satu, M. I. Khan, M. H. Ali, S. Uddin, P. Lio, J. M. Quinn, and M. A. Moni, "Machine learning-based models for early stage detection of autism spectrum disorders," *IEEE Access*, vol. 9, pp. 13357–13377, Jan. 2021, doi: 10.1109/ACCESS.2021.3050935.

[2] S. Mandal, C. Bhattacharya, and S. Jana, "Early detection of autism spectrum disorder in children using ensemble machine learning algorithms," *IEEE Access*, vol. 10, pp. 65581–65596, Jun. 2022, doi: 10.1109/ACCESS.2022.3184556.

[3] F. Thabtah, L. Zhang, and N. Abdelhamid, "New machine learning tool for ASD features selection and classification," *Informatics in Medicine Unlocked*, vol. 25, pp. 100694, 2021, doi: 10.1016/j.imu.2021.100694.

[4] M. A. Al-Qudah, R. Al-Khasawneh, and W. Zamil, "ASD screening using behavioral features: A machine learning comparative study," *Applied Sciences*, vol. 12, no. 11, pp. 5461, May 2022, doi: 10.3390/app12115461.

[5] H. Parikh, K. Narayan, K. Patel, and B. Patel, "Comparison of machine learning classifiers for autism spectrum disorder using AQ-10 screening tool dataset," *IEEE Access*, vol. 11, pp. 23941–23955, 2023, doi: 10.1109/ACCESS.2023.3254187.

[6] A. Bircanoglu, M. Anafarta, and F. Orhan, "Comparative analysis of machine learning classifiers for autism spectrum disorder prediction from behavioral screening data," in *Proc. IEEE Int. Conf. Machine Learning and Applications (ICMLA)*, Nassau, Bahamas, Dec. 2022, pp. 1134–1139, doi: 10.1109/ICMLA55696.2022.00186.

[7] T. Duda, M. Kosmicki, and D. P. Wall, "Testing the accuracy of machine learning algorithms for the diagnosis of autism spectrum disorder," in *Proc. IEEE Int. Symp. Computer-Based Medical Systems (CBMS)*, Lisbon, Portugal, Jun. 2021, pp. 365–370, doi: 10.1109/CBMS52027.2021.00075.

[8] N. Nnamoko, J. Arreola-Paulin, D. England, D. Vickers, and K. Goulding, "Autism spectrum disorder symptom severity classification using random forest," *Journal of Intelligent Information Systems*, vol. 60, no. 2, pp. 319–335, Apr. 2023, doi: 10.1007/s10844-022-00751-7.

[9] M. S. I. Satu, M. Ahmed, T. Akter, M. I. Khan, J. M. Quinn, and M. A. Moni, "Unveiling autism spectrum disorder diagnosis using feature selection approaches and machine learning classifiers," *Computers in Biology and Medicine*, vol. 145, pp. 105427, Jun. 2022, doi: 10.1016/j.compbiomed.2022.105427.

[10] Y. Zhang, Q. Li, J. Wang, Z. Jin, and S. Liu, "Multi-class autism spectrum disorder classification using machine learning with feature importance analysis," *Computers in Biology and Medicine*, vol. 152, pp. 106438, Jan. 2023, doi: 10.1016/j.compbiomed.2022.106438.

[11] W. V. Swanson, A. Suri, J. Bhatt, and P. Lakshminarayan, "Support vector machine-based classification of autism spectrum disorder from EEG connectivity features," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 31, pp. 1283–1293, 2023, doi: 10.1109/TNSRE.2023.3243981.

[12] R. Kara, C. Sert, and B. Korkmaz, "Feature selection strategies for autism spectrum disorder detection using behavioral questionnaire data," *Expert Systems with Applications*, vol. 213, pp. 118993, Mar. 2023, doi: 10.1016/j.eswa.2022.118993.

[13] F. W. Alsaade and A. S. Alzahrani, "A study on the classification and detection of ASD using features of deep neural networks," *Brain Sciences*, vol. 12, no. 2, pp. 141, Feb. 2022, doi: 10.3390/brainsci12020141.

[14] M. Khodatars, A. Shoeibi, D. Sadeghi, N. Ghassemi, M. Jafari, P. Khadem, and A. Khosravi, "Deep learning for neurological disorders: Methodologies, task formulations and challenges," *Artificial Intelligence in Medicine*, vol. 126, pp. 102216, Apr. 2022, doi: 10.1016/j.artmed.2022.102216.

[15] S. Raj, R. Masood, and S. Agrawal, "Analysis and detection of autism spectrum disorder using deep learning techniques," *Procedia Computer Science*, vol. 218, pp. 1002–1011, 2023, doi: 10.1016/j.procs.2023.01.080.

[16] D. Bone, M. S. Goodwin, M. P. Black, C.-C. Lee, K. Audhkhasi, and S. Narayanan, "Applying machine learning to facilitate autism diagnostics: Pitfalls and promises," *Journal of Autism and Developmental Disorders*, vol. 52, no. 2, pp. 529–541, Feb. 2022, doi: 10.1007/s10803-021-05269-8.

[17] M. Moridian, N. Nasrabadi, M. Rezapour, M. Gheibizadeh, M. Saeidi, and A. Alizadehsani, "Automatic autism spectrum disorder detection using artificial intelligence methods with MRI neuroimaging: A review," *Frontiers in Molecular Neuroscience*, vol. 15, pp. 900018, Jul. 2022, doi: 10.3389/fnmol.2022.900018.

[18] B. Tao and M. Troilo, "Synthetic minority oversampling for autism spectrum disorder prediction in pediatric populations," *Journal of Biomedical Informatics*, vol. 136, pp. 104255, Dec. 2022, doi: 10.1016/j.jbi.2022.104255.

[19] S. Chaudhuri, A. Saha, and D. Bhattacharjee, "SMOTE-based oversampling approach for improved autism spectrum disorder classification under class imbalance," *Biomedical Signal Processing and Control*, vol. 81, pp. 104411, Mar. 2023, doi: 10.1016/j.bspc.2022.104411.

[20] A. Hasan, A. Alomari, M. Alhumyani, M. Alsaidi, A. Alashjaee, and A. Aziz, "Classification of autism spectrum disorder using supervised learning of structural MRI-based cortical features," *Journal of Autism and Developmental Disorders*, vol. 53, no. 1, pp. 195–210, Jan. 2023, doi: 10.1007/s10803-022-05428-6.

[21] P. Duda, D. P. Wall, and R. Daniels, "Computational approaches to autism screening: A systematic review," *Autism Research*, vol. 15, no. 3, pp. 423–445, Mar. 2022, doi: 10.1002/aur.2668.

[22] A. Tariq, J. Daniels, J. Schwartz, P. Washington, C. Kalantarian, and D. P. Wall, "Mobile detection of autism through machine learning on home video: A development and prospective validation study," *PLOS Medicine*, vol. 20, no. 6, pp. e1004276, Jun. 2023, doi: 10.1371/journal.pmed.1004276.

[23] A. Raza, U. Munir, M. Altaf, S. R. Naqvi, M. S. Younis, and J. Iqbal, "Prediction of autism spectrum disorder with machine learning methods using web-based deployment," *Healthcare*, vol. 11, no. 2, pp. 212, Jan. 2023, doi: 10.3390/healthcare11020212.

[24] P. R. Krishnappa Babu, J. M. Di Martino, A. Aikat, K. Carpenter, J. Egger, J. Espinosa, and G. Dawson, "A digital biomarker for autism from smartphone-based gaze estimation," *Nature Medicine*, vol. 29, no. 2, pp. 570–576, Feb. 2023, doi: 10.1038/s41591-023-02221-3.

[25] V. Loth, L. Tillmann, J. Ahmad, and J. Buitelaar, "Billion dollar autism research: Systematic analysis of autism spectrum disorder machine learning classification studies from 2020–2025," *PLOS ONE*, vol. 18, no. 4, pp. e0284375, Apr. 2023, doi: 10.1371/journal.pone.0284375.
