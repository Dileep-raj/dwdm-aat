import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the dataset
sales_data = pd.read_csv("online_sales_data.csv")
num_itemsets = 10

# =============================
# Apriori Algorithm Implementation
# =============================
# Create a pivot table for products and transactions
basket = pd.pivot_table(
    sales_data,
    values="Units Sold",
    index="Region",  # Using Region as a proxy for transactions
    columns="Product Category",
    aggfunc="sum",
    fill_value=0
)

# Encode data to binary format
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Apply Apriori
frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True)
rules_apriori = association_rules(
    frequent_itemsets, num_itemsets, metric="lift", min_threshold=1.2)

# Plot Top 10 Rules by Lift
top_rules = rules_apriori.nlargest(10, 'lift')
plt.figure(figsize=(10, 6))
plt.barh(top_rules['consequents'].astype(str),
         top_rules['lift'], color='skyblue')
plt.xlabel("Lift")
plt.ylabel("Rules")
plt.title("Apriori")
plt.show()

# =============================
# FP-Growth Algorithm Implementation
# =============================
# Apply FP-Growth
frequent_itemsets_fp = fpgrowth(basket, min_support=0.1, use_colnames=True)
rules_fpgrowth = association_rules(
    frequent_itemsets_fp, num_itemsets, metric="lift", min_threshold=1.0)

# Plot Top 10 Rules by Confidence
top_fp_rules = rules_fpgrowth.nlargest(10, 'confidence')
plt.figure(figsize=(10, 6))
plt.barh(top_fp_rules['consequents'].astype(str),
         top_fp_rules['confidence'], color='orange')
plt.xlabel("Confidence")
plt.ylabel("Rules")
plt.title("FP-Growth")
plt.show()

# =============================
# DBSCAN Clustering Implementation
# =============================
# Select numerical features for clustering
features = sales_data[['Units Sold', 'Total Revenue', 'Product Category']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
sales_data['Cluster'] = dbscan.fit_predict(scaled_features)

# Visualize DBSCAN Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=sales_data['Product Category'],
    y=sales_data['Total Revenue'],
    hue=sales_data['Cluster'],
    palette='viridis',
    legend='full'
)
plt.title("DBSCAN Clustering")
plt.xlabel("Revenue")
plt.ylabel("Profit Margin")
plt.show()

# =============================
# Save Results to CSV (Optional)
# =============================
frequent_itemsets.to_csv("apriori_frequent_itemsets.csv", index=False)
rules_apriori.to_csv("apriori_rules.csv", index=False)
frequent_itemsets_fp.to_csv("fpgrowth_frequent_itemsets.csv", index=False)
rules_fpgrowth.to_csv("fpgrowth_rules.csv", index=False)
sales_data.to_csv("dbscan_clusters.csv", index=False)
