from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Synthetische data genereren voor demo
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# K-means clustering trainen
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Plot aanmaken
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', edgecolors='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.title('K-Means Clustering')
plt.legend()
plt.show()
