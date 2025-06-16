import matplotlib
matplotlib.use('Agg') # Set the backend to 'Agg' BEFORE importing pyplot

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix

# Import ONLY the best performing model (MLPClassifier) and LightGBM for comparison
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb # Keeping LightGBM for the 'origin' feature impact and its potential

# Import evaluation metrics and plotting tools
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, make_scorer
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
FILE_PATH_1 = 'name_gender_dataset.csv'
FILE_PATH_2 = 'ethiopian_names_dataset.csv'
NAME_COLUMN = 'name'
GENDER_COLUMN = 'gender'

NGRAM_RANGE = (2, 4)
MAX_NGRAM_FEATURES = 7000
USE_TFIDF = True

# --- Function to evaluate and plot results ---
# This function is kept for consistency in reporting the final tuned model's performance
def evaluate_and_plot_model(model, X_data, y_true, model_name, gender_labels, store_results_dict):
    y_pred = model.predict(X_data)
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_data)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_data)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    roc_auc = 0.0
    fpr, tpr = [0, 1], [0, 1]
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
        except ValueError as e:
            print(f"Warning: Could not compute ROC AUC for {model_name} - {e}")
            roc_auc = np.nan

    store_results_dict[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC': roc_auc
    }

    print(f"\n--- Results for {model_name} ---")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC: {roc_auc:.4f}")
    print(f"  Classification Report:\n{classification_report(y_true, y_pred, target_names=gender_labels)}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=gender_labels, yticklabels=gender_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(f'{model_name.replace(" ", "_")}_confusion_matrix.png')
    plt.close()

    # Plot ROC Curve
    if y_prob is not None and not np.isnan(roc_auc):
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) for {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(f'{model_name.replace(" ", "_")}_roc_curve.png')
        plt.close()

# --- Step 1: Data Collection/Loading ---
print("--- Step 1: Data Collection/Loading ---")
print(f"Loading data from {FILE_PATH_1}...")
try:
    df1 = pd.read_csv(FILE_PATH_1)
    print(f"Loaded {len(df1)} rows from {FILE_PATH_1}.")
except FileNotFoundError:
    print(f"Error: {FILE_PATH_1} not found. Please check the path.")
    exit()

print(f"Loading data from {FILE_PATH_2}...")
try:
    df2 = pd.read_csv(FILE_PATH_2)
    print(f"Loaded {len(df2)} rows from {FILE_PATH_2}.")
except FileNotFoundError:
    print(f"Error: {FILE_PATH_2} not found. Please check the path.")
    exit()

required_columns = [NAME_COLUMN, GENDER_COLUMN]
if not all(col in df1.columns for col in required_columns):
    print(f"Error: {FILE_PATH_1} must contain '{NAME_COLUMN}' and '{GENDER_COLUMN}' columns.")
    exit()
if not all(col in df2.columns for col in required_columns):
    print(f"Error: {FILE_PATH_2} must contain '{NAME_COLUMN}' and '{GENDER_COLUMN}' columns.")
    exit()

df1['origin'] = 'General'
df2['origin'] = 'Ethiopian'

df_combined = pd.concat([
    df1[[NAME_COLUMN, GENDER_COLUMN, 'origin']],
    df2[[NAME_COLUMN, GENDER_COLUMN, 'origin']]
], ignore_index=True)

print(f"Combined DataFrame has {len(df_combined)} rows.")
print("Combined Data Head:\n", df_combined.head())

# --- Step 2: Preprocessing ---
print("\n--- Step 2: Preprocessing ---")

print("Performing text normalization...")
df_combined[NAME_COLUMN] = df_combined[NAME_COLUMN].astype(str).str.lower().str.strip()
df_combined[NAME_COLUMN] = df_combined[NAME_COLUMN].str.replace(r'[^a-z\s]', '', regex=True)

initial_rows = len(df_combined)
df_combined.dropna(subset=[NAME_COLUMN, GENDER_COLUMN], inplace=True)
df_combined = df_combined[df_combined[NAME_COLUMN].str.len() > 0]
if len(df_combined) < initial_rows:
    print(f"Removed {initial_rows - len(df_combined)} rows with missing or empty names after cleaning.")
print(f"DataFrame after cleaning: {len(df_combined)} rows.")

print("Encoding gender labels...")
label_encoder = LabelEncoder()
df_combined['gender_encoded'] = label_encoder.fit_transform(df_combined[GENDER_COLUMN])
gender_labels = label_encoder.classes_
print(f"Gender Mappings: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

print("Creating features (Name Length, First/Last Letter)...")
df_combined['name_length'] = df_combined[NAME_COLUMN].apply(len)
df_combined['first_letter'] = df_combined[NAME_COLUMN].str[0]
df_combined['last_letter'] = df_combined[NAME_COLUMN].str[-1]

# --- Step 3: Data Splitting ---
print("\n--- Step 3: Data Splitting ---")
X = df_combined[[NAME_COLUMN, 'name_length', 'first_letter', 'last_letter', 'origin']]
y = df_combined['gender_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set size: {len(X_train)} rows")
print(f"Test set size: {len(X_test)} rows")

print(f"Vectorizing names using character N-grams (TF-IDF: {USE_TFIDF})...")
if USE_TFIDF:
    char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=NGRAM_RANGE, max_features=MAX_NGRAM_FEATURES)
else:
    char_vectorizer = CountVectorizer(analyzer='char', ngram_range=NGRAM_RANGE, max_features=MAX_NGRAM_FEATURES)

X_train_char_features = char_vectorizer.fit_transform(X_train[NAME_COLUMN])
X_test_char_features = char_vectorizer.transform(X_test[NAME_COLUMN])
print(f"N-gram features (train) shape: {X_train_char_features.shape}")
print(f"N-gram features (test) shape: {X_test_char_features.shape}")

print("One-Hot Encoding first and last letter features...")
ohe_features_train = pd.get_dummies(X_train[['first_letter', 'last_letter']])
ohe_features_test = pd.get_dummies(X_test[['first_letter', 'last_letter']])
ohe_features_test = ohe_features_test.reindex(columns=ohe_features_train.columns, fill_value=0)
print(f"One-Hot Encoded (first/last) features (train) shape: {ohe_features_train.shape}")
print(f"One-Hot Encoded (first/last) features (test) shape: {ohe_features_test.shape}")

print("One-Hot Encoding 'origin' feature...")
ohe_origin_train = pd.get_dummies(X_train['origin'], prefix='origin')
ohe_origin_test = pd.get_dummies(X_test['origin'], prefix='origin')
ohe_origin_test = ohe_origin_test.reindex(columns=ohe_origin_train.columns, fill_value=0)
print(f"One-Hot Encoded (origin) features (train) shape: {ohe_origin_train.shape}")
print(f"One-Hot Encoded (origin) features (test) shape: {ohe_origin_test.shape}")

print("Combining all features...")
X_train_length_feature = X_train['name_length'].values.reshape(-1, 1)
X_test_length_feature = X_test['name_length'].values.reshape(-1, 1)

X_train_final = hstack([X_train_char_features, ohe_features_train.values, ohe_origin_train.values, X_train_length_feature])
X_test_final = hstack([X_test_char_features, ohe_features_test.values, ohe_origin_test.values, X_test_length_feature])

# Convert final sparse matrices to CSR format (still important for LightGBM if you choose to include it later)
X_train_final_csr = csr_matrix(X_train_final)
X_test_final_csr = csr_matrix(X_test_final)

print(f"Final Training Features (X_train_final) shape: {X_train_final.shape}")
print(f"Final Testing Features (X_test_final) shape: {X_test_final.shape}")


# --- Step 4: Hyperparameter Tuning for the Best Performing Model (MLPClassifier) ---
print("\n--- Step 4: Hyperparameter Tuning for the Best Performing Model (MLPClassifier) ---")

tuned_models_results = {} # To store results of the tuned model

# MLPClassifier Tuning (your previous best performing model)
print("\n--- Tuning Simple Neural Network (MLP) ---")
mlp_param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50), (200, 100), (200,)], # Experiment with different network structures
    'alpha': [0.0001, 0.001, 0.01], # L2 regularization strength
    'learning_rate_init': [0.001, 0.005, 0.01],
    'max_iter': [700, 1000] # Increased max_iter for more robust convergence
}
mlp_grid_search = GridSearchCV(
    MLPClassifier(random_state=42, solver='adam', early_stopping=True, validation_fraction=0.1, verbose=False),
    mlp_param_grid, cv=3, scoring=make_scorer(f1_score), verbose=1, n_jobs=-1
)
mlp_grid_search.fit(X_train_final, y_train)
print(f"Best MLP parameters: {mlp_grid_search.best_params_}")
print(f"Best MLP F1-score (on validation sets): {mlp_grid_search.best_score_:.4f}")
best_mlp_tuned = mlp_grid_search.best_estimator_


# --- Step 5: Final Evaluation of the Best Tuned Model ---
print("\n--- Step 5: Final Evaluation of the Best Tuned Model ---")

print("\nEvaluating Tuned Simple Neural Network (MLP) on the test set...")
evaluate_and_plot_model(best_mlp_tuned, X_test_final, y_test, "Tuned MLPClassifier", gender_labels, tuned_models_results)

# --- Final Model Performance Summary ---
print("\n--- Final Model Performance Summary ---")
final_results_df = pd.DataFrame(tuned_models_results).T
print("Comprehensive FINAL Tuned Model Performance:")
print(final_results_df.round(4))

final_best_accuracy = final_results_df.loc["Tuned MLPClassifier", 'Accuracy']
final_best_f1 = final_results_df.loc["Tuned MLPClassifier", 'F1-Score']

print(f"\nThe TUNED MLPClassifier achieved an Accuracy of: **{final_best_accuracy:.4f}**")
print(f"And an F1-Score of: **{final_best_f1:.4f}**")


print("\n--- Pipeline Complete ---")
print("Targeting 90%+ accuracy: Review the results for the Tuned MLPClassifier.")
print("If still not at 90% accuracy, consider these options:")
print("  1. **Refine the MLP `mlp_param_grid`:** Add more hidden layer configurations, or a wider range for `alpha` or `learning_rate_init`.")
print("  2. **Expand N-gram Features:** Try `NGRAM_RANGE = (1, 5)` or `MAX_NGRAM_FEATURES = 10000`.")
print("  3. **Data Augmentation:** If possible, acquire and integrate more diverse name datasets.")
print("  4. **Advanced Neural Networks:** For significant jumps, exploring specialized architectures like Character-level CNNs or LSTMs could be beneficial, but this involves switching to deep learning frameworks (TensorFlow/PyTorch).")