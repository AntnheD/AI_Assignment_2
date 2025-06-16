import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load and Combine Data
# -----------------------------
print("Loading datasets...")
df1 = pd.read_csv('ethiopian_names_dataset.csv')  # Replace with your actual dataset path
df2 = pd.read_csv('name_gender_dataset.csv')

# Concatenate both datasets
df = pd.concat([df1, df2], ignore_index=True)

# Basic info
print("\nFirst 5 rows:")
print(df.head())
print("\nColumns:", df.columns.tolist())

# -----------------------------
# Step 2: General Statistics
# -----------------------------
total_entries = df.shape[0]
unique_names = df['name'].nunique()
duplicate_rows = df[df.duplicated(['name', 'gender'], keep=False)].shape[0]

print(f"\nTotal entries: {total_entries}")
print(f"Unique names: {unique_names}")
print(f"Duplicate (name + gender) entries: {duplicate_rows}")

# Gender distribution
print("\nGender distribution (%):")
print(df['gender'].value_counts(normalize=True) * 100)

# -----------------------------
# Step 3: Fairness Evaluation
# -----------------------------
# Plot gender distribution
print("\nPlotting gender distribution...")
df['gender'].value_counts().plot(kind='bar', title='Gender Distribution', color='skyblue')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Check ambiguous names (names used for multiple genders)
print("\nChecking ambiguous names...")
name_gender_counts = df.groupby('name')['gender'].nunique()
ambiguous_names = name_gender_counts[name_gender_counts > 1]
print(f"Number of names used for multiple genders: {len(ambiguous_names)}")

# Optional: Show some examples
print("\nTop 10 ambiguous names:")
print(ambiguous_names.sort_values(ascending=False).head(10))

# -----------------------------
# Step 4: Name Characteristics
# -----------------------------
print("\nAnalyzing name characteristics...")

# Add new features
df['name_length'] = df['name'].str.len()
df['last_letter'] = df['name'].str[-1].str.lower()
df['first_letter'] = df['name'].str[0].str.lower()

# Average name length
avg_length = df['name_length'].mean()
print(f"Average name length: {avg_length:.2f}")

# Most common last letters
print("\nMost common last letters:")
print(df['last_letter'].value_counts().head(10))

# Most common first letters
print("\nMost common first letters:")
print(df['first_letter'].value_counts().head(10))