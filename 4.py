import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
file_path = r'C:\Users\merij\Desktop\Dataset KennisKluis\Gebied_07\Vondsten\07_vondsten_lijst.csv'
df = pd.read_csv(file_path, sep=';')

# Clean numeric columns with commas and convert to float
numeric_cols = df.select_dtypes(include=['object']).columns
df[numeric_cols] = df[numeric_cols].apply(lambda x: pd.to_numeric(x.str.replace(',', ''), errors='coerce'))

# Dummy (one-hot encode) string variables
columns_to_dummify = ['FASE', 'Spoordatering_begin', 'Spoordatering_eind', 'Categorie ABR', 'Grondstof', 'Voorwerp', 'Functie', 'Opmerkingen', 'Groep', 'Hoofdgroep']
df = pd.get_dummies(df, columns=columns_to_dummify)

# Display basic statistics of the dataset
print(df.describe())

# Create a pair plot for visualizing relationships between variables
sns.pairplot(df)
plt.show()

# Create a heatmap matrix to visualize correlation between variables
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()