from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

sales_data = pd.read_csv('online_sales_data.csv')

# Step 1: Prepare the data for Apriori
# Create a basket matrix with Region as proxy for transactions and Product Name as items
basket = pd.pivot_table(
    sales_data,
    values="Units Sold",
    index="Region",
    columns="Product Category",
    aggfunc="sum",
    fill_value=0
)

# Convert the basket to a binary matrix
basket_binary = basket.map(lambda x: 1 if x > 0 else 0)

# Step 2: Apply Apriori algorithm
frequent_itemsets_apriori = apriori(basket_binary, min_support=0.01, use_colnames=True)

# Generate association rules
rules_apriori = association_rules(frequent_itemsets_apriori,num_itemsets=2, metric="lift", min_threshold=1.2)

# Display the top results
print(frequent_itemsets_apriori.head())
print(rules_apriori.head())

# Create a Treemap
fig = px.treemap(
    sales_data,
    path=['Region', 'Product Category'],  # Hierarchical levels: Region -> Product Category
    values='Total Revenue',              # Size of the blocks
    color='Total Revenue',               # Color based on Total Revenue
    title="Treemap of Total Revenue by Region and Product Category",
    color_continuous_scale='Viridis'
)
fig.show()

# Visualization: Top 10 rules by lift
top_rules = rules_apriori.nlargest(10, 'lift')
plt.barh(top_rules['consequents'].astype(str), top_rules['lift'], color='skyblue')
plt.xlabel("Lift")
plt.ylabel("Rules")
plt.title("Apriori")
plt.show()