from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Gebruik slechts 2 waardes voor binary classification
X, y = X[y != 2], y[y != 2]

# Split de dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train de Logistic Regression classifier
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Maak voorspellingen met de test subset
y_pred = log_reg.predict(X_test)

# Maak een confusion matrix aan
conf_matrix = confusion_matrix(y_test, y_pred)

# Maak plot aan
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names[:2], yticklabels=iris.target_names[:2])
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
