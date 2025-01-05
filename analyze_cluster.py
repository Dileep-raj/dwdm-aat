import pandas as pd

# Load the dataset with clusters
df = pd.read_csv("sales_data_with_clusters.csv")

# Group by the cluster and calculate key metrics
cluster_analysis = df.groupby('Cluster').agg(
    Total_Revenue=('Revenue', 'sum'),
    Total_Profit=('Profit Margin', 'sum'),
    Average_Quantity_Sold=('Quantity Sold', 'mean'),
    Customer_Count=('Customer Age Group', 'count')  # Count of rows per cluster
).reset_index()

# Sort clusters by Total Revenue
cluster_analysis = cluster_analysis.sort_values(by='Total_Revenue', ascending=False)

# Display the results
print("Cluster Analysis:")
print(cluster_analysis)

# Save the analysis to a CSV file for reference
cluster_analysis.to_csv("cluster_analysis.csv", index=False)
print("Cluster analysis saved to 'cluster_analysis.csv'")
