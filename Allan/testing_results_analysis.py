import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Helper function to calculate the score based on sector and multiplier
def calculate_score(sector, multiplier):
    return sector * multiplier

# Function to calculate the angular distance between sectors
def calculate_angular_distance(actual_sector, detected_sector):
    # Each sector is 18° apart
    sector_angle = 18  # Degrees per sector
    actual_angle = (actual_sector - 1) * sector_angle
    detected_angle = (detected_sector - 1) * sector_angle
    angular_distance = abs(actual_angle - detected_angle)
    # Ensure the distance is no more than 180° (half the dartboard)
    return min(angular_distance, 360 - angular_distance)

if __name__ == '__main__':
    # Load the Excel file
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'CapstoneTestData.xlsx')
    
    # Load both sheets
    actual_scores = pd.read_excel(file_path, sheet_name='TEST 1 Actual Scores')
    detected_scores = pd.read_excel(file_path, sheet_name='TEST 1 Detected Scores')

    # Print column names for verification
    print("Actual Scores Columns:", actual_scores.columns)
    print("Detected Scores Columns:", detected_scores.columns)

    # Calculate actual scores for each dart
    actual_scores['Actual_Dart1'] = actual_scores.apply(lambda row: calculate_score(row['Dart 1 sector '], row['dart 1 multiplier']), axis=1)
    actual_scores['Actual_Dart2'] = actual_scores.apply(lambda row: calculate_score(row['Dart 2 sector '], row['dart 2 multiplier']), axis=1)
    actual_scores['Actual_Dart3'] = actual_scores.apply(lambda row: calculate_score(row['Dart 3 sector'], row['Dart 3 multiplier']), axis=1)

    # Rename columns in detected_scores for consistency
    detected_scores.rename(columns={'Dart1': 'Dart1_Detected', 'Dart2': 'Dart2_Detected', 'Dart3': 'Dart3_Detected'}, inplace=True)

    # Convert detected scores to numeric, coercing errors to NaN
    detected_scores['Dart1_Detected'] = pd.to_numeric(detected_scores['Dart1_Detected'], errors='coerce')
    detected_scores['Dart2_Detected'] = pd.to_numeric(detected_scores['Dart2_Detected'], errors='coerce')
    detected_scores['Dart3_Detected'] = pd.to_numeric(detected_scores['Dart3_Detected'], errors='coerce')

    # Merge actual and detected scores
    merged_data = actual_scores.merge(
        detected_scores[['Trial #', 'Dart1_Detected', 'Dart2_Detected', 'Dart3_Detected']], on='Trial #')

    # Combine the data into a single column for analysis
    merged_data_melted = merged_data.melt(id_vars=['Trial #'], value_vars=['Actual_Dart1', 'Actual_Dart2', 'Actual_Dart3', 'Dart1_Detected', 'Dart2_Detected', 'Dart3_Detected'],
                                          var_name='Dart Type', value_name='Score')
    
    # Split the melted data into actual and detected columns
    merged_data_melted['Type'] = merged_data_melted['Dart Type'].apply(lambda x: 'Actual' if 'Actual' in x else 'Detected')
    merged_data_melted['Dart Number'] = merged_data_melted['Dart Type'].apply(lambda x: x.split('_')[1] if '_' in x else x.split(' ')[1])

    # Pivot the data to have Actual and Detected scores side by side for each trial and dart
    merged_data_pivoted = merged_data_melted.pivot_table(index=['Trial #', 'Dart Number'], columns='Type', values='Score', aggfunc='first').reset_index()

    # Apply the angular distance calculation for actual vs detected scores
    merged_data_pivoted['Angular_Error'] = merged_data_pivoted.apply(
        lambda row: calculate_angular_distance(row['Actual'], row['Detected']),
        axis=1
    )

    # Display the angular errors
    print(merged_data_pivoted[['Trial #', 'Dart Number', 'Angular_Error']].head())

    # Plot the angular errors
    plt.figure(figsize=(12, 6))
    sns.histplot(merged_data_pivoted['Angular_Error'], kde=True, color='purple', bins=20)
    plt.xlabel('Angular Error (Degrees)')
    plt.ylabel('Frequency')
    plt.title('Angular Error Distribution for Detected vs Actual Scores (All Darts)')
    plt.show()
