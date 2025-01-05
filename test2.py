import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
sales_data = pd.read_csv("online_sales_data.csv")

# Step 1: Prepare numerical data for clustering
# Select relevant numerical features for clustering
features = sales_data[['Units Sold', 'Total Revenue']]

# Scale the features for DBSCAN
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 2: Apply DBSCAN
# Adjust `eps` and `min_samples` as needed
dbscan = DBSCAN(eps=0.5, min_samples=5)
sales_data['Cluster'] = dbscan.fit_predict(scaled_features)

# Step 3: Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=sales_data['Units Sold'],
    y=sales_data['Total Revenue'],
    hue=sales_data['Cluster'],
    palette='viridis',
    legend='full'
)
plt.title("DBSCAN Clustering: Units Sold vs Total Revenue")
plt.xlabel("Units Sold")
plt.ylabel("Total Revenue")
plt.legend(title="Cluster", loc='upper left', bbox_to_anchor=(1, 1))
plt.grid()
plt.show()

# Step 4: Summary of clusters
print("Cluster Summary:")
print(sales_data['Cluster'].value_counts())