import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack # For combining sparse matrices

# --- Configuration ---
FILE_PATH_1 = 'ethiopian_names_dataset.csv' # Path to the first CSV file
FILE_PATH_2 = 'name_gender_dataset.csv' # Path to the second CSV file
NAME_COLUMN = 'name'    # Column name for the person's name
GENDER_COLUMN = 'gender' # Column name for the gender label

# Define the range for character N-grams.
# (2, 4) means bigrams, trigrams, and quadrigrams will be generated.
NGRAM_RANGE = (2, 4)
MAX_NGRAM_FEATURES = 5000 # Limit the number of N-gram features to avoid too many dimensions

# --- Step 1: Load Data from Both CSV Files ---
print(f"Loading data from {FILE_PATH_1}...")
try:
    df1 = pd.read_csv(FILE_PATH_1)
    print(f"Loaded {len(df1)} rows from {FILE_PATH_1}.")
except FileNotFoundError:
    print(f"Error: {FILE_PATH_1} not found. Please check the path.")
    exit() # Exit if file is not found

print(f"\nLoading data from {FILE_PATH_2}...")
try:
    df2 = pd.read_csv(FILE_PATH_2)
    print(f"Loaded {len(df2)} rows from {FILE_PATH_2}.")
except FileNotFoundError:
    print(f"Error: {FILE_PATH_2} not found. Please check the path.")
    exit() # Exit if file is not found

# --- Step 2: Combine DataFrames ---
# Ensure both dataframes have the required columns before concatenating
required_columns = [NAME_COLUMN, GENDER_COLUMN]
if not all(col in df1.columns for col in required_columns):
    print(f"Error: {FILE_PATH_1} must contain '{NAME_COLUMN}' and '{GENDER_COLUMN}' columns.")
    exit()
if not all(col in df2.columns for col in required_columns):
    print(f"Error: {FILE_PATH_2} must contain '{NAME_COLUMN}' and '{GENDER_COLUMN}' columns.")
    exit()

# Select only the relevant columns and concatenate
df_combined = pd.concat([
    df1[[NAME_COLUMN, GENDER_COLUMN]],
    df2[[NAME_COLUMN, GENDER_COLUMN]]
], ignore_index=True)

print(f"\nCombined DataFrame has {len(df_combined)} rows.")
print("Combined Data Head:\n", df_combined.head())

# --- Step 3: Text Normalization ---
print("\nPerforming text normalization...")
# Convert names to lowercase and remove leading/trailing whitespace
df_combined[NAME_COLUMN] = df_combined[NAME_COLUMN].str.lower().str.strip()

# Optional: Remove non-alphabetic characters (e.g., numbers, punctuation)
# This keeps only letters and spaces (if names like "Mary Ann" are present)
df_combined[NAME_COLUMN] = df_combined[NAME_COLUMN].str.replace(r'[^a-z\s]', '', regex=True)

# Remove any rows where the name became empty after cleaning
initial_rows = len(df_combined)
df_combined = df_combined[df_combined[NAME_COLUMN].str.len() > 0]
if len(df_combined) < initial_rows:
    print(f"Removed {initial_rows - len(df_combined)} rows with empty names after cleaning.")

print("Normalized Data Head:\n", df_combined.head())

# --- Step 4: Handle Missing Data (if any) ---
print("\nChecking for missing values...")
print(df_combined.isnull().sum())
# Drop rows where either name or gender is missing
df_combined.dropna(subset=[NAME_COLUMN, GENDER_COLUMN], inplace=True)
print(f"DataFrame after dropping rows with missing values: {len(df_combined)} rows.")


# --- Step 5: Label Encode Gender (Target Variable) ---
print("\nEncoding gender labels...")
label_encoder = LabelEncoder()
df_combined['gender_encoded'] = label_encoder.fit_transform(df_combined[GENDER_COLUMN])

# Display mappings
gender_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(f"Gender Mappings: {gender_mapping}")


# --- Step 6: Feature Engineering ---
print("\nCreating features (Name Length, First/Last Letter, N-grams)...")

# Feature 1: Name Length
df_combined['name_length'] = df_combined[NAME_COLUMN].apply(len)

# Feature 2: First Letter
df_combined['first_letter'] = df_combined[NAME_COLUMN].str[0]

# Feature 3: Last Letter
df_combined['last_letter'] = df_combined[NAME_COLUMN].str[-1]

# --- Step 7: Data Splitting (BEFORE N-gram Vectorization to prevent data leakage) ---
# It's crucial to fit the vectorizer ONLY on the training data.
X = df_combined[[NAME_COLUMN, 'name_length', 'first_letter', 'last_letter']]
y = df_combined['gender_encoded']

# Stratify by gender to ensure equal proportions in train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# You might want a validation set too, depending on your model tuning needs:
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
# (0.25 of 0.8 is 0.2, so it makes it 60/20/20 train/val/test)

print(f"\nTrain set size: {len(X_train)} rows")
print(f"Test set size: {len(X_test)} rows")


# --- Step 8: Encode Categorical Features (One-Hot & N-grams) ---

# N-gram Feature Extraction
# Using CountVectorizer. TfidfVectorizer is another excellent option.
char_vectorizer = CountVectorizer(
    analyzer='char',
    ngram_range=NGRAM_RANGE,
    max_features=MAX_NGRAM_FEATURES # Limit the vocabulary size
)

# Fit the vectorizer on the TRAINING data names ONLY, then transform both train and test
X_train_char_features = char_vectorizer.fit_transform(X_train[NAME_COLUMN])
X_test_char_features = char_vectorizer.transform(X_test[NAME_COLUMN]) # Transform, not fit_transform!

print(f"N-gram features (train) shape: {X_train_char_features.shape}")
print(f"N-gram features (test) shape: {X_test_char_features.shape}")


# One-Hot Encode First and Last Letter features
# Combine train and test for OHE to ensure consistent columns, then split back
# Or use OneHotEncoder and fit only on train
ohe_features_train = pd.get_dummies(X_train[['first_letter', 'last_letter']])
ohe_features_test = pd.get_dummies(X_test[['first_letter', 'last_letter']])

# Ensure test set has same columns as train set (important for OHE)
# Use reindex with columns from the training set's one-hot encoded features
ohe_features_test = ohe_features_test.reindex(columns=ohe_features_train.columns, fill_value=0)

print(f"One-Hot Encoded features (train) shape: {ohe_features_train.shape}")
print(f"One-Hot Encoded features (test) shape: {ohe_features_test.shape}")


# --- Step 9: Combine All Features ---
# Convert name_length from Pandas Series to NumPy array and reshape for hstack
X_train_length_feature = X_train['name_length'].values.reshape(-1, 1)
X_test_length_feature = X_test['name_length'].values.reshape(-1, 1)

# Horizontally stack all feature matrices for training set
X_train_final = hstack([
    X_train_char_features,
    ohe_features_train.values,
    X_train_length_feature
])

# Horizontally stack all feature matrices for testing set
X_test_final = hstack([
    X_test_char_features,
    ohe_features_test.values,
    X_test_length_feature
])

print("\n--- Preprocessing Complete ---")
print(f"Final Training Features (X_train_final) shape: {X_train_final.shape}")
print(f"Final Training Labels (y_train) shape: {y_train.shape}")
print(f"Final Testing Features (X_test_final) shape: {X_test_final.shape}")
print(f"Final Testing Labels (y_test) shape: {y_test.shape}")

print("\nYour data is now preprocessed and ready for training a classification model!")
# You can now use X_train_final, y_train, X_test_final, y_test with models like
# LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, etc.