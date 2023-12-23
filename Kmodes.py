import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler

# Laad dataset vanuit excel, drop nans
file_path = '/Users/BerenddeJeu/Desktop/WeTransfer Danille/Gebied_07/TBL_Vullingen.xlsx'
df = pd.read_excel(file_path)
df = df.dropna(subset=['Diepte'])

# Print een head om een beeld van de structuur te krijgen
print(df.head())

# Kwantitatieve en kwalitatieve data scheiden.
numerical_cols = df.select_dtypes(include=['int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# dummificeren
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Standardiseren
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# indices bepalen
numerical_indices = list(range(len(numerical_cols)))
categorical_indices = list(range(len(numerical_cols), len(df_encoded.columns)))

# Initialiseer K-prototypes
kproto = KPrototypes(n_clusters=3, init='Cao', n_init=3, verbose=2, random_state=42)

# Model in elkaar zetten
clusters = kproto.fit_predict(df_encoded.values, categorical=categorical_indices)

# Cluster labels
df['Cluster'] = clusters

# Cluster verkennen
cluster_summary = df.groupby('Cluster').mean()
print(cluster_summary)
 
