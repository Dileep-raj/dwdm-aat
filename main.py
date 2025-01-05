import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
sales_data = pd.read_csv("online_sales_data.csv")

# Step 1: Prepare data for clustering
# Aggregate data: Sum up Units Sold for each Product Category in each Region
cluster_data = sales_data.groupby(['Region', 'Product Category'])['Units Sold'].sum().reset_index()

# Pivot the table to have Product Categories as columns and Regions as rows
cluster_matrix = cluster_data.pivot(index='Region', columns='Product Category', values='Units Sold').fillna(0)

# Step 2: Scale the data
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(cluster_matrix)

# Step 3: Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust the number of clusters as needed
cluster_labels = kmeans.fit_predict(scaled_matrix)

# Add cluster labels to the original data
cluster_matrix['Cluster'] = cluster_labels

# Step 4: Visualize clustering results
plt.figure(figsize=(10, 6))
sns.heatmap(
    cluster_matrix.drop(columns=['Cluster']),
    cmap='viridis',
    annot=True,
    fmt=".1f",
    linewidths=0.5,
    cbar=True
)
plt.title("Heatmap of Units Sold by Product Category and Region")
plt.show()

# Step 5: Plot Regions in clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=scaled_matrix[:, 0],  # First principal feature
    y=scaled_matrix[:, 1],  # Second principal feature
    hue=cluster_labels,
    palette='viridis',
    s=100
)
plt.title("KMeans Clustering of Regions")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
plt.legend(title="Cluster")
plt.show()

# Print Cluster Summary
cluster_summary = cluster_matrix.groupby('Cluster').sum()
print("Cluster Summary:")
print(cluster_summary)


# Aggregate total revenue by region
region_revenue = sales_data.groupby('Region')['Total Revenue'].sum().reset_index()

# Sort data for better visualization
region_revenue = region_revenue.sort_values(by='Total Revenue', ascending=False)

# Plot a bar chart for total revenue by region
plt.figure(figsize=(10, 6))
sns.barplot(data=region_revenue, x='Region', y='Total Revenue', palette='Blues_d')

# Add annotations for the total revenue values
for index, row in region_revenue.iterrows():
    plt.text(index, row['Total Revenue'] + 0.02 * max(region_revenue['Total Revenue']), 
             f"${row['Total Revenue']:.2f}", ha='center', fontsize=10)

# Title and labels
plt.title("Total Revenue by Region", fontsize=16)
plt.xlabel("Region", fontsize=12)
plt.ylabel("Total Revenue", fontsize=12)
plt.show()