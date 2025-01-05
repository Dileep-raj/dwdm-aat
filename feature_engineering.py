import pandas as pd

# Load the dataset
df = pd.read_csv("sales_data.csv")

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Create new time-based features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.day_name()

# Display the first few rows to verify
print(df.head())
