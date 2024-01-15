import pandas as pd

# Load the CSV file into a pandas DataFrame with semicolon as separator
df = pd.read_csv(r'C:\Users\merij\Desktop\Dataset KennisKluis\Gebied_07\Vondsten\07_vondsten_lijst.csv', sep=';')

# Drop specified columns
columns_to_drop = ['Groep', 'Hoofdgroep', 'Opmerkingen']
df = df.drop(columns=columns_to_drop)

# Display information about missing values in the original DataFrame
print("Original DataFrame - Missing Values:")
print(df.isnull().sum())

# Remove rows with NaN values
df_cleaned = df.dropna()

# Display the first few rows of the cleaned DataFrame
print("\nCleaned DataFrame:")
print(df_cleaned.head())
