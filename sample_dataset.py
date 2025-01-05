import pandas as pd
import random
import numpy as np

# Define the parameters
regions = ['North', 'South', 'East', 'West']
products = ['Electronics', 'Apparel', 'Furniture', 'Groceries']
date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

# Generate synthetic data
data = {
    'Date': random.choices(date_range, k=5000),  # Random 1000 dates
    'Region': random.choices(regions, k=5000),
    'Product': random.choices(products, k=5000),
    'Quantity Sold': [random.randint(1, 100) for _ in range(5000)],  # Random quantities
    'Revenue': [round(random.uniform(10, 500), 2) for _ in range(5000)],  # Random revenue
    'Profit Margin': [round(random.uniform(5, 30), 2) for _ in range(5000)],  # Random profit margin
    'Customer Age Group': random.choices(['18-25', '26-35', '36-50', '50+'], k=5000),
    'Customer Gender': random.choices(['Male', 'Female'], k=5000)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('sales_data.csv', index=False)

print("Synthetic dataset generated and saved as 'sales_data.csv'")
