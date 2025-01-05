# kmeans_clustering.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('sales_data.csv')

# Select numerical features for clustering
features = ['Quantity Sold', 'Revenue', 'Profit Margin']

# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize Clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['Revenue'], df['Profit Margin'],
            c=df['Cluster'], cmap='viridis', s=50)
plt.colorbar(label='Cluster')
plt.title('Clusters Based on Revenue and Profit Margin')
plt.xlabel('Revenue')
plt.ylabel('Profit Margin')
plt.grid()
plt.show()

# Save the cluster assignments
df.to_csv('sales_data_with_clusters.csv', index=False)
print("Clusters saved to 'sales_data_with_clusters.csv'")
