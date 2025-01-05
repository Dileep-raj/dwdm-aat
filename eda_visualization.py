import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('sales_data.csv')

# Create subplots: let's say you want a 2x2 grid for 4 charts
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Chart 1
sns.barplot(x='Region', y='Revenue', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Revenue by Region')

# Chart 2
sns.boxplot(x='Region', y='Profit Margin', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Profit Margin by Region')

# Chart 3
sns.histplot(df['Revenue'], kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Distribution of Revenue')

# Chart 4
sns.countplot(x='Product', data=df, ax=axes[1, 1])
axes[1, 1].set_title('Count of Products')

# Adjust layout for better spacing
plt.tight_layout()

# Show all charts in one window
plt.show()
