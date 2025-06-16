import pandas as pd
import numpy as np # Often useful for random selections

def balance_gender_proportion(df):
    """
    Removes excess names from the majority gender to achieve approximate fairness.

    Args:
        df (pd.DataFrame): The input DataFrame with 'name' and 'gender' columns.

    Returns:
        pd.DataFrame: The DataFrame with balanced gender proportions.
    """

    print("--- Initial Dataset Info ---")
    print(f"Total entries: {len(df)}")
    print("Initial Gender distribution (%):")
    print(df['gender'].value_counts(normalize=True) * 100)
    print("\n" + "="*40 + "\n")

    # Ensure no exact (name, gender) duplicates remain (though your previous run showed none)
    df_cleaned = df.drop_duplicates(subset=['name', 'gender'], keep='first')
    if len(df) != len(df_cleaned):
        print(f"Removed {len(df) - len(df_cleaned)} exact (name, gender) duplicates.")
    df = df_cleaned # Use the deduplicated DataFrame for balancing

    female_df = df[df['gender'] == 'F']
    male_df = df[df['gender'] == 'M']

    num_female = len(female_df)
    num_male = len(male_df)

    print(f"Current Female names: {num_female}")
    print(f"Current Male names: {num_male}")

    if num_female == num_male:
        print("\nGender distribution is already balanced. No removals needed.")
        return df
    elif num_female > num_male:
        gender_to_reduce = 'F'
        excess_count = num_female - num_male
        df_to_reduce = female_df
        df_to_keep = male_df
        target_count = num_male
    else: # num_male > num_female
        gender_to_reduce = 'M'
        excess_count = num_male - num_female
        df_to_reduce = male_df
        df_to_keep = female_df
        target_count = num_female

    print(f"\n{gender_to_reduce} names are in the majority. Need to remove {excess_count} entries.")
    print(f"Target count for both genders: {target_count} entries each.")

    # Randomly sample without replacement to keep only the target count
    # .sample() returns a new DataFrame with the sampled rows
    df_reduced_sampled = df_to_reduce.sample(n=target_count, random_state=42) # random_state for reproducibility

    # Concatenate the reduced majority gender with the minority gender
    df_balanced = pd.concat([df_reduced_sampled, df_to_keep])

    print("\n--- After Balancing Gender Proportions ---")
    print(f"Total entries after balancing: {len(df_balanced)}")
    print("Gender distribution (%):")
    print(df_balanced['gender'].value_counts(normalize=True) * 100)
    print("\n" + "="*40 + "\n")

    return df_balanced

# --- Example Usage ---

# 1. Load your dataset
try:
    df = pd.read_csv('name_gender_dataset.csv')  # Replace with your actual dataset path
    # If your columns are different, rename them here:
    # df.rename(columns={'YourNameCol': 'name', 'YourGenderCol': 'gender'}, inplace=True)
except FileNotFoundError:
    print("Error: 'ethiopian_names_dataset.csv' not found. Creating a dummy DataFrame for demonstration.")
    data = {
        'name': ['Makda', 'Samuel', 'Kidus', 'Kifle', 'Genet', 'Sarah', 'Alex', 'Fatima', 'Aisha', 'Chika', 'Kwame', 'Nzinga', 'Desta', 'Yohannes', 'Abebe'],
        'gender': ['F', 'M', 'M', 'M', 'F', 'F', 'M', 'F', 'F', 'F', 'M', 'F', 'F', 'M', 'M'],
        'count': [100, 200, 50, 70, 120, 150, 80, 90, 110, 60, 40, 130, 70, 100, 95],
        'Probability': [0.9, 0.95, 0.8, 0.7, 0.92, 0.98, 0.6, 0.85, 0.9, 0.75, 0.5, 0.88, 0.72, 0.81, 0.78]
    }
    df = pd.DataFrame(data)

# Process the DataFrame to balance gender
balanced_df = balance_gender_proportion(df.copy()) # Use .copy() to avoid modifying the original DataFrame

# Save the balanced dataset to a new CSV file
balanced_df.to_csv('name_gender_dataset.csv', index=False)
print("Balanced dataset saved to 'ethiopian_names_dataset_balanced.csv'")

# Optional: Display first few rows of the balanced data
print("\nFirst 5 rows of the balanced dataset:")
print(balanced_df.head())

# Optional: Final check of counts from the saved file
# test_df = pd.read_csv('ethiopian_names_dataset_balanced.csv')
# print("\nFinal check from saved CSV:")
# print(test_df['gender'].value_counts())