# Libraries importeren
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dataset inladen van sklearn
diabetes = load_diabetes()
X = diabetes.data[:, np.newaxis, 2]  # Voor gemak gebruik ik slechts 2 var's
y = diabetes.target

# Split de dataset in een training en test gedeelte
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start Lin-Reg model
linear_reg_model = LinearRegression()

# Train het model met trainingsdata
linear_reg_model.fit(X_train, y_train)

# Maak voorspellingen
y_pred = linear_reg_model.predict(X_test)

# Prestatie beoordelen en evalueren
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Lineare regressielijn plotten
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('Linear Regression: Diabetes Dataset')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.show()
