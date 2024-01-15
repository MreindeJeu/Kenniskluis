import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(r'C:\Users\merij\Desktop\Dataset KennisKluis\Gebied_07\Vondsten\USE.csv')

# Display the column names to check for any typos or inconsistencies
print("Original Column Names:")
print(df.columns)

# Specify columns to drop
columns_to_drop = ['User', 'Groep', 'Hoofdgroep', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Opmerkingen']

# Drop specified columns
df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

# Drop rows with NaN values only in specified columns
df_cleaned = df_cleaned.dropna(subset=['Voorwerp', 'Aantal fragmenten', 'Spoordatering_begin', 'Spoordatering_eind'])

# Dummy encode categorical variables
categorical_columns = ['FASE', 'Categorie ABR', 'Grondstof', 'Voorwerp', 'Functie']
df_cleaned = pd.get_dummies(df_cleaned, columns=categorical_columns, prefix=categorical_columns)

# Convert dummy variables to binary (1 for true, 0 for false)
df_cleaned = df_cleaned.astype(int)

# Fit k-means clustering model using 'Spoordatering_begin' and 'Aantal fragmenten'
data_for_clustering_begin = df_cleaned[['Spoordatering_begin', 'Aantal fragmenten']]
kmeans_begin = KMeans(n_clusters=3, random_state=42)
df_cleaned['Cluster_begin'] = kmeans_begin.fit_predict(data_for_clustering_begin)

# Fit k-means clustering model using 'Spoordatering_eind' and 'Aantal fragmenten'
data_for_clustering_eind = df_cleaned[['Spoordatering_eind', 'Aantal fragmenten']]
kmeans_eind = KMeans(n_clusters=3, random_state=42)
df_cleaned['Cluster_eind'] = kmeans_eind.fit_predict(data_for_clustering_eind)

# Fit k-means clustering model using 'Spoordatering_begin' and 'Categorie ABR_KER'
data_for_clustering_cat = df_cleaned[['Spoordatering_begin', 'Categorie ABR_KER']]
kmeans_cat = KMeans(n_clusters=3, random_state=42)
df_cleaned['Cluster_cat'] = kmeans_cat.fit_predict(data_for_clustering_cat)

# Fit k-means clustering model using 'Spoordatering_eind' and 'Categorie ABR_KER'
data_for_clustering_cat_eind = df_cleaned[['Spoordatering_eind', 'Categorie ABR_KER']]
kmeans_cat_eind = KMeans(n_clusters=3, random_state=42)
df_cleaned['Cluster_cat_eind'] = kmeans_cat_eind.fit_predict(data_for_clustering_cat_eind)

# Display the first few rows of the DataFrame with cluster labels
print("\nDataFrame with Cluster Labels (Spoordatering_begin):")
print(df_cleaned.head())

# Set seaborn style for improved aesthetics
sns.set(style="whitegrid")

plt.figure(figsize=(8, 8))
sns.scatterplot(x='Spoordatering_begin', y='Categorie ABR_KER', hue='Cluster_cat', palette='viridis', data=df_cleaned, legend='full')
plt.scatter(kmeans_cat.cluster_centers_[:, 0], kmeans_cat.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')  # Highlight centroids
plt.title('K-Means Clustering with Centroids (Spoordatering_begin and Categorie ABR_KER)')
plt.legend()
plt.xlabel('Spoordatering (Years)')
plt.ylabel('Categorie ABR_KER (Binary)')
plt.tight_layout()
plt.show()

# Plot 2D scatter plot of the clustered data with 'Spoordatering_eind' and 'Categorie ABR_KER'
plt.figure(figsize=(8, 8))
sns.scatterplot(x='Spoordatering_eind', y='Categorie ABR_KER', hue='Cluster_cat_eind', palette='viridis', data=df_cleaned, legend='full')
plt.scatter(kmeans_cat_eind.cluster_centers_[:, 0], kmeans_cat_eind.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')  # Highlight centroids
plt.title('K-Means Clustering with Centroids (Spoordatering_eind and Categorie ABR_KER)')
plt.legend()
plt.xlabel('Spoordatering (Years)')
plt.ylabel('Categorie ABR_KER (Binary)')
plt.tight_layout()
plt.show()