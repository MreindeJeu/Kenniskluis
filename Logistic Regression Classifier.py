from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Data inladen
file_path = '/Users/BerenddeJeu/Desktop/WeTransfer Danille/Gebied_07/TBL_Vullingen.xlsx'

# Excel -> Pandas
df = pd.read_excel(file_path)

# Kies kolommen voor X en Y
features = ['Spoor', 'PUT', 'VLAK', 'VORM', 'CONTOUR', 'AARD', 'OPMERKING', 'Vulling']
target = 'Diepte'
df = df.dropna(subset=['Diepte'])

# Set Y
y = df[target]

# Set X
X = df[features]

# One-hot encoding
X = pd.get_dummies(X, columns=['Spoor', 'PUT', 'VLAK', 'VORM', 'CONTOUR', 'AARD', 'OPMERKING', 'Vulling'])

# Split de dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Voorspelling doen met test set
y_pred = linear_reg.predict(X_test)

# Evalueren aan de hand van Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
