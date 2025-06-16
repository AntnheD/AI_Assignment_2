import pandas as pd

def process_names_dataset(df):
    """
    Removes duplicate (name, gender) entries and checks for equal gender proportion.

    Args:
        df (pd.DataFrame): The input DataFrame with at least 'name' and 'gender' columns.

    Returns:
        pd.DataFrame: The DataFrame after removing duplicates.
    """

    print("--- Initial Dataset Info ---")
    print(f"Total entries: {len(df)}")
    print(f"Unique names (before duplicate removal): {df['name'].nunique()}")
    print("Initial Gender distribution (%):")
    print(df['gender'].value_counts(normalize=True) * 100)
    print("\n" + "="*40 + "\n")

    # 1. Remove exact duplicate (name, gender) entries
    # The 'subset' argument ensures we consider both 'name' and 'gender' columns
    # for identifying duplicates. keep='first' means it keeps the first occurrence.
    df_cleaned = df.drop_duplicates(subset=['name', 'gender'], keep='first')

    print("--- After Removing Exact (name, gender) Duplicates ---")
    print(f"Total entries after removing duplicates: {len(df_cleaned)}")
    print(f"Unique names: {df_cleaned['name'].nunique()}") # Unique names might stay similar if duplicates were mostly exact pairs

    # 2. Check male and female proportions after cleaning
    gender_counts = df_cleaned['gender'].value_counts()
    male_count = gender_counts.get('M', 0)
    female_count = gender_counts.get('F', 0)
    total_cleaned_entries = len(df_cleaned)

    if total_cleaned_entries == 0:
        print("\nNo entries remaining after cleaning. Cannot calculate proportions.")
        return df_cleaned

    male_proportion = (male_count / total_cleaned_entries) * 100
    female_proportion = (female_count / total_cleaned_entries) * 100

    print("\nGender distribution (%) after removing exact (name, gender) duplicates:")
    print(f"Male: {male_proportion:.2f}% ({male_count} entries)")
    print(f"Female: {female_proportion:.2f}% ({female_count} entries)")

    print("\n--- Fairness Check ---")
    if male_count == female_count:
        print("Male and Female names are in exactly equal proportion.")
    else:
        print("Male and Female names are NOT in equal proportion.")
        if female_count > male_count:
            difference = female_count - male_count
            print(f"There are {difference} more Female names than Male names.")
            print(f"To achieve fairness (equal counts), you would need to remove approximately {difference} Female names.")
            print(f"The target count for both genders would be {male_count} entries each.")
        else: # male_count > female_count
            difference = male_count - female_count
            print(f"There are {difference} more Male names than Female names.")
            print(f"To achieve fairness (equal counts), you would need to remove approximately {difference} Male names.")
            print(f"The target count for both genders would be {female_count} entries each.")

    return df_cleaned

# --- Example Usage ---

# 1. Load your dataset
# Replace 'ethiopian_names_dataset.csv' with the actual path to your file
try:
    df = pd.read_csv('ethiopian_names_dataset.csv' and 'name_gender_dataset.csv')
    # Ensure column names are correct if they are different in your CSV
    # df.rename(columns={'NameColumn': 'name', 'GenderColumn': 'gender'}, inplace=True)
except FileNotFoundError:
    print("Error: 'ethiopian_names_dataset.csv' not found. Please make sure the file is in the same directory or provide the full path.")
    # Create a dummy DataFrame for demonstration if file not found
    print("\nCreating a dummy DataFrame for demonstration purposes:")
    data = {
        'name': ['Makda', 'Samuel', 'Kidus', 'Kifle', 'Genet', 'Makda', 'Alex', 'Alex', 'Sarah', 'John', 'John', 'Sarah'],
        'gender': ['F', 'M', 'M', 'M', 'F', 'F', 'M', 'F', 'F', 'M', 'M', 'F'],
        'count': [100, 200, 50, 70, 120, 100, 80, 75, 150, 250, 250, 160],
        'Probability': [0.9, 0.95, 0.8, 0.7, 0.92, 0.9, 0.6, 0.65, 0.98, 0.99, 0.99, 0.97]
    }
    df = pd.DataFrame(data)

# Process the DataFrame
cleaned_df = process_names_dataset(df.copy()) # Use .copy() to avoid modifying the original DataFrame