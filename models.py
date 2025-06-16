import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

# Import the models you want to test
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC # Often better for large, sparse data than SVC(kernel='linear')
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb # Make sure to 'pip install lightgbm'
from sklearn.neural_network import MLPClassifier # Simple Feedforward Neural Network

# Import evaluation metrics and plotting tools
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# IMPORTANT: Update these paths to your actual CSV file locations
FILE_PATH_1 = 'name_gender_dataset.csv'
FILE_PATH_2 = 'ethiopian_names_dataset.csv'
NAME_COLUMN = 'name'    # Ensure this matches your name column header
GENDER_COLUMN = 'gender' # Ensure this matches your gender column header

# Preprocessing parameters
NGRAM_RANGE = (2, 4) # Character n-grams from 2 to 4 characters
MAX_NGRAM_FEATURES = 5000 # Limits the n-gram vocabulary size to top 5000 features

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

# Ensure both dataframes have the required columns before concatenating
required_columns = [NAME_COLUMN, GENDER_COLUMN]
if not all(col in df1.columns for col in required_columns):
    print(f"Error: {FILE_PATH_1} must contain '{NAME_COLUMN}' and '{GENDER_COLUMN}' columns.")
    exit()
if not all(col in df2.columns for col in required_columns):
    print(f"Error: {FILE_PATH_2} must contain '{NAME_COLUMN}' and '{GENDER_COLUMN}' columns.")
    exit()

# Combine DataFrames
df_combined = pd.concat([
    df1[[NAME_COLUMN, GENDER_COLUMN]],
    df2[[NAME_COLUMN, GENDER_COLUMN]]
], ignore_index=True)

print(f"Combined DataFrame has {len(df_combined)} rows.")
print("Combined Data Head:\n", df_combined.head())

# --- Step 2: Preprocessing ---
print("\n--- Step 2: Preprocessing ---")

# Text Normalization
print("Performing text normalization...")
df_combined[NAME_COLUMN] = df_combined[NAME_COLUMN].astype(str).str.lower().str.strip() # Convert to string first
df_combined[NAME_COLUMN] = df_combined[NAME_COLUMN].str.replace(r'[^a-z\s]', '', regex=True)

# Handle Missing Data and Empty Names
initial_rows = len(df_combined)
df_combined.dropna(subset=[NAME_COLUMN, GENDER_COLUMN], inplace=True)
df_combined = df_combined[df_combined[NAME_COLUMN].str.len() > 0]
if len(df_combined) < initial_rows:
    print(f"Removed {initial_rows - len(df_combined)} rows with missing or empty names after cleaning.")
print(f"DataFrame after cleaning: {len(df_combined)} rows.")

# Label Encode Gender (Target Variable)
print("Encoding gender labels...")
label_encoder = LabelEncoder()
df_combined['gender_encoded'] = label_encoder.fit_transform(df_combined[GENDER_COLUMN])
gender_labels = label_encoder.classes_ # Get original labels in order (e.g., ['Female', 'Male'])
print(f"Gender Mappings: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Feature Engineering
print("Creating features (Name Length, First/Last Letter)...")
df_combined['name_length'] = df_combined[NAME_COLUMN].apply(len)
df_combined['first_letter'] = df_combined[NAME_COLUMN].str[0]
df_combined['last_letter'] = df_combined[NAME_COLUMN].str[-1]

# --- Step 3: Data Splitting ---
print("\n--- Step 3: Data Splitting ---")
X = df_combined[[NAME_COLUMN, 'name_length', 'first_letter', 'last_letter']]
y = df_combined['gender_encoded']

# Split data into training and testing sets
# stratify=y ensures that both train and test sets have roughly the same proportion of genders as the full dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set size: {len(X_train)} rows")
print(f"Test set size: {len(X_test)} rows")

# N-gram Feature Extraction (fit on train, transform on test)
print("Vectorizing names using character N-grams...")
char_vectorizer = CountVectorizer(analyzer='char', ngram_range=NGRAM_RANGE, max_features=MAX_NGRAM_FEATURES)
X_train_char_features = char_vectorizer.fit_transform(X_train[NAME_COLUMN])
X_test_char_features = char_vectorizer.transform(X_test[NAME_COLUMN])
print(f"N-gram features (train) shape: {X_train_char_features.shape}")
print(f"N-gram features (test) shape: {X_test_char_features.shape}")

# One-Hot Encode First and Last Letter features
print("One-Hot Encoding first and last letter features...")
ohe_features_train = pd.get_dummies(X_train[['first_letter', 'last_letter']])
ohe_features_test = pd.get_dummies(X_test[['first_letter', 'last_letter']])

# Ensure test set has the same columns as train set (critical for OHE consistency)
ohe_features_test = ohe_features_test.reindex(columns=ohe_features_train.columns, fill_value=0)
print(f"One-Hot Encoded features (train) shape: {ohe_features_train.shape}")
print(f"One-Hot Encoded features (test) shape: {ohe_features_test.shape}")

# Combine all features for training and testing
print("Combining all features...")
X_train_length_feature = X_train['name_length'].values.reshape(-1, 1)
X_test_length_feature = X_test['name_length'].values.reshape(-1, 1)

X_train_final = hstack([X_train_char_features, ohe_features_train.values, X_train_length_feature])
X_test_final = hstack([X_test_char_features, ohe_features_test.values, X_test_length_feature])

print(f"Final Training Features (X_train_final) shape: {X_train_final.shape}")
print(f"Final Testing Features (X_test_final) shape: {X_test_final.shape}")

# --- Step 4: Model Development and Evaluation ---
print("\n--- Step 4: Model Development and Evaluation ---")

# Define the models to test
models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42, max_iter=2000),
    "Linear SVM (SVC)": LinearSVC(random_state=42, max_iter=2000),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "LightGBM": lgb.LGBMClassifier(random_state=42, n_jobs=-1),
    "Simple Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, solver='adam',
                                                 learning_rate='adaptive', early_stopping=True, validation_fraction=0.1,
                                                 verbose=False)
}

results = {} # To store evaluation metrics for comparison

# Function to evaluate and plot results for a single model
def evaluate_and_plot_model(model, X_test, y_test, model_name, gender_labels):
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1] # Probability of the positive class (e.g., Female)
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_test) # Raw scores for models like LinearSVC

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # ROC AUC calculation (only if y_prob is available and model provides valid scores)
    roc_auc = 0.0
    fpr, tpr = [0, 1], [0, 1] # Default for plotting if AUC can't be computed
    if y_prob is not None and len(np.unique(y_test)) > 1: # Ensure there are at least two classes in y_test for ROC
        try:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
        except ValueError as e:
            print(f"Warning: Could not compute ROC AUC for {model_name} - {e}")
            roc_auc = np.nan # Assign NaN if AUC calculation fails

    results[model_name] = {
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
    print(f"  Classification Report:\n{classification_report(y_test, y_pred, target_names=gender_labels)}")


    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=gender_labels, yticklabels=gender_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show() # <--- THIS IS KEY: Ensures the plot is displayed before moving on

    # Plot ROC Curve (only if y_prob is available)
    if y_prob is not None and not np.isnan(roc_auc): # Only plot if AUC was successfully computed
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) for {model_name}')
        plt.legend(loc="lower right")
        plt.show() # <--- THIS IS KEY: Ensures the plot is displayed before moving on

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    try:
        model.fit(X_train_final, y_train)
        evaluate_and_plot_model(model, X_test_final, y_test, name, gender_labels)
    except Exception as e:
        print(f"Error training or evaluating {name}: {e}")
        results[name] = {'Accuracy': np.nan, 'Precision': np.nan, 'Recall': np.nan, 'F1-Score': np.nan, 'AUC': np.nan}


# --- Step 5: Model Comparison Summary ---
print("\n--- Step 5: Model Comparison Summary ---")
results_df = pd.DataFrame(results).T # Transpose to have models as rows
results_df = results_df.sort_values(by='F1-Score', ascending=False) # Sort by F1-Score

print("Comprehensive Model Performance Comparison (Sorted by F1-Score):")
print(results_df.round(4))

# --- Choosing the Best Model ---
best_model_name = results_df.index[0]
print(f"\nBased on F1-Score, the best performing model is: **{best_model_name}**")
print(f"It achieved an F1-Score of: **{results_df.loc[best_model_name, 'F1-Score']:.4f}**")

print("\n--- Testing Complete ---")
print("You can now proceed with hyperparameter tuning for the selected model(s) or deploy the best one.")