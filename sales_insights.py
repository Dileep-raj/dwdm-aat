import pandas as pd

# Load the data
df = pd.read_csv('sales_data_with_clusters.csv')

# Key Insights 1: Top-performing Products
top_products = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head(5)
print("Top 5 Products by Revenue:\n", top_products)

# Key Insights 2: Total Revenue by Region
revenue_by_region = df.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
print("\nTotal Revenue by Region:\n", revenue_by_region)

# Key Insights 3: Average Quantity Sold and Profit Margin by Region
avg_sales_by_region = df.groupby('Region')[['Quantity Sold', 'Profit Margin']].mean()
print("\nAverage Sales and Profit Margin by Region:\n", avg_sales_by_region)

# Key Insights 4: Seasonality - Trends over Time
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
monthly_sales = df.groupby('Month')['Revenue'].sum()

print("\nMonthly Sales Trend:\n", monthly_sales)

# Actionable Strategies: Example recommendations based on findings
recommendations = """
Recommendations:
1. Focus marketing efforts on the top-performing products such as 'Apparel' and 'Electronics' to boost revenue.
2. Invest in underperforming regions (e.g., South) to improve market share.
3. Consider launching new promotions during months with lower sales (e.g., January) to increase revenue.
4. Leverage customer demographics insights to target specific age groups for personalized marketing.
"""
print("\nActionable Recommendations:\n", recommendations)

# Save the insights to a CSV file for later reporting
insights = pd.DataFrame({
    "Insight Type": ["Top Products", "Revenue by Region", "Avg Sales by Region", "Monthly Sales Trend", "Recommendations"],
    "Details": [top_products.to_string(), revenue_by_region.to_string(), avg_sales_by_region.to_string(), monthly_sales.to_string(), recommendations]
})

insights.to_csv('sales_insights_report.csv', index=False)
