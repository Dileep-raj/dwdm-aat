import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset
df = pd.read_csv('sales_data.csv')

# Ensure the Date column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Feature Engineering: Extract Year, Month, and Day of the week from the Date column
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Select target variable and features
target = 'Revenue'
features = ['Product', 'Region', 'Quantity Sold', 'Customer Age Group', 'Year', 'Month', 'DayOfWeek']

# Define the features (X) and target (y)
X = df[features]
y = df[target]

# One-hot encoding for categorical variables (Product, Region, Customer Age Group)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Product', 'Region', 'Customer Age Group']),
        ('num', 'passthrough', ['Quantity Sold', 'Year', 'Month', 'DayOfWeek'])
    ])

# Create a pipeline with preprocessing and Linear Regression model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')
