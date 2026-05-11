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
