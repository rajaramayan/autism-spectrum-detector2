# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, log_loss)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE

MODELS_DIR = "models"

# ==========================================
# 2. LOAD DATASET
# ==========================================
df = pd.read_csv("Toddler Autism dataset July 2018.csv")
df.drop(columns=['Case_No'], inplace=True)

print("Shape:", df.shape)
print(df.info())
print(df.describe())

# ==========================================
# 3. PREPROCESSING
# ==========================================
df.drop_duplicates(inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Encode categorical variables (keep per-column encoders)
le_dict = {}
df_encoded = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    le_dict[col] = le

for col in df_encoded.columns:
    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
df_encoded.fillna(df_encoded.mean(), inplace=True)

# Split features and target
X = df_encoded.iloc[:, :-1]
y = df_encoded.iloc[:, -1]
feature_names = X.columns.tolist()

# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 5. HANDLE IMBALANCE (SMOTE)
# ==========================================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ==========================================
# 6. FEATURE SCALING
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 7. DEFINE MODELS (same config as Streamlit app)
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_split=20, min_samples_leaf=10,
        ccp_alpha=0.005, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=15,
        min_samples_leaf=6, max_features='sqrt',
        min_impurity_decrease=0.001, random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=51, weights='uniform', metric='minkowski', p=1),
    "SVM (Poly)": SVC(kernel='poly', degree=2, C=0.1, gamma='scale', probability=True),
    "SVM (RBF)": SVC(kernel='rbf', C=0.1, gamma='scale', probability=True),
    "Naive Bayes": GaussianNB(var_smoothing=1e-8),
    "QDA": QuadraticDiscriminantAnalysis(reg_param=0.7)
}

# ==========================================
# 8. TRAIN + EVALUATE CLASSICAL MODELS
# ==========================================
results = []
roc_data = []
trained_models = {}

columns = ["Model", "Train Accuracy", "Accuracy", "Gap",
           "Precision", "Recall", "Specificity", "F1 Score", "ROC-AUC", "Log Loss"]

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model

    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_prob)
    logloss  = log_loss(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    gap = train_acc - acc

    results.append([name, train_acc, acc, gap, prec, rec, specificity, f1, roc_auc, logloss])

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data.append((name, fpr, tpr, roc_auc))

# ==========================================
# 9. ANN MODEL (MLPClassifier — same as Streamlit)
# ==========================================
print("\nTraining ANN (MLPClassifier)...")
ann = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42
)
ann.fit(X_train_scaled, y_train)

train_acc_ann = accuracy_score(y_train, ann.predict(X_train_scaled))
y_prob_ann = ann.predict_proba(X_test_scaled)[:, 1]
y_pred_ann = ann.predict(X_test_scaled)

acc  = accuracy_score(y_test, y_pred_ann)
prec = precision_score(y_test, y_pred_ann)
rec  = recall_score(y_test, y_pred_ann)
f1   = f1_score(y_test, y_pred_ann)
roc_auc  = roc_auc_score(y_test, y_prob_ann)
logloss  = log_loss(y_test, y_prob_ann)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ann).ravel()
specificity = tn / (tn + fp)
gap_ann = train_acc_ann - acc

results.append(["ANN", train_acc_ann, acc, gap_ann, prec, rec, specificity, f1, roc_auc, logloss])

fpr_ann, tpr_ann, _ = roc_curve(y_test, y_prob_ann)
roc_data.append(("ANN", fpr_ann, tpr_ann, roc_auc))

# ==========================================
# 10. CREATE RESULTS DATAFRAME
# ==========================================
results_df = pd.DataFrame(results, columns=columns)

# Rank by ROC-AUC, then by explicit model priority when tied.
# ANN ranks first among tied models (best soft-probability calibration);
# Decision Tree ranks last (outputs hard 0/1 probabilities — overconfident).
MODEL_PRIORITY = {
    "ANN": 1, "Random Forest": 2, "Logistic Regression": 3,
    "SVM (RBF)": 4, "QDA": 5, "KNN": 6,
    "Naive Bayes": 7, "SVM (Poly)": 8, "Decision Tree": 9
}
results_df["_priority"] = results_df["Model"].map(MODEL_PRIORITY).fillna(10)
results_df_sorted = results_df.sort_values(
    by=["ROC-AUC", "_priority"], ascending=[False, True]
).drop(columns=["_priority"]).reset_index(drop=True)

print("\nFinal Model Comparison:\n")
print(results_df_sorted.to_string(index=False))

# ==========================================
# 11. OVERFITTING ANALYSIS
# ==========================================
def gap_status(gap):
    if gap < 0.02:   return "No overfitting"
    if gap < 0.05:   return "Mild — acceptable"
    if gap < 0.10:   return "Moderate — needs justification"
    return "Severe overfitting"

gap_df = results_df_sorted[["Model", "Train Accuracy", "Accuracy", "Gap"]].copy()
gap_df["Status"] = gap_df["Gap"].apply(gap_status)
print("\nOverfitting Analysis:\n")
print(gap_df.to_string(index=False))

# ==========================================
# 12. ROC CURVE (ALL MODELS)
# ==========================================
line_styles = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-']
colors = plt.cm.tab10(np.linspace(0, 0.9, 9))
fpr_grid = np.linspace(0, 1, 300)

plt.figure(figsize=(10, 8))
for i, (name, fpr, tpr, auc) in enumerate(roc_data):
    # Keep only first occurrence of each FPR value so the curve starts at (0,0)
    _, first_idx = np.unique(fpr, return_index=True)
    tpr_smooth = np.interp(fpr_grid, fpr[first_idx], tpr[first_idx])
    plt.plot(fpr_grid, tpr_smooth,
             label=f"{name} (AUC={auc:.4f})",
             linewidth=2,
             linestyle=line_styles[i % len(line_styles)],
             color=colors[i])
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (All Models Including ANN)")
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 13. BAR PLOT — PERFORMANCE COMPARISON
# ==========================================
fig, ax = plt.subplots(figsize=(12, 6))
results_df_sorted.set_index("Model")[["Accuracy", "F1 Score", "ROC-AUC"]].plot(kind='bar', ax=ax)
ax.set_title("Model Performance Comparison (All Models)")
ax.set_ylabel("Score")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ==========================================
# 14. CONFUSION MATRIX (BEST MODEL)
# ==========================================
best_model_name = results_df_sorted.iloc[0]["Model"]

if best_model_name == "ANN":
    y_pred_best = y_pred_ann
else:
    y_pred_best = trained_models[best_model_name].predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix — {best_model_name}")
plt.tight_layout()
plt.show()

# ==========================================
# 15. FINAL BEST MODEL SUMMARY
# ==========================================
print("\nBEST MODEL:")
print(results_df_sorted.iloc[0])

# ==========================================
# 16. SAVE ALL MODELS & ARTIFACTS TO DISK
# ==========================================
os.makedirs(MODELS_DIR, exist_ok=True)

with open(os.path.join(MODELS_DIR, "trained_models.pkl"), "wb") as f:
    pickle.dump(trained_models, f)

with open(os.path.join(MODELS_DIR, "ann.pkl"), "wb") as f:
    pickle.dump(ann, f)

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODELS_DIR, "le_dict.pkl"), "wb") as f:
    pickle.dump(le_dict, f)

with open(os.path.join(MODELS_DIR, "results_df.pkl"), "wb") as f:
    pickle.dump(results_df, f)

with open(os.path.join(MODELS_DIR, "roc_data.pkl"), "wb") as f:
    pickle.dump(roc_data, f)

metadata = {
    "feature_names": feature_names,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
}
with open(os.path.join(MODELS_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print(f"\nAll models and artifacts saved to '{MODELS_DIR}/' folder.")
print("   You can now run the Streamlit app -- it will load these pre-trained models.")