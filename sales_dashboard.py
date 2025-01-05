import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load the sales data
df = pd.read_csv('sales_data.csv')

# Aggregate data for insights
top_products = df.groupby('Product')['Revenue'].sum().nlargest(5).reset_index()
revenue_by_region = df.groupby('Region')['Revenue'].sum().reset_index()
avg_sales_profit_by_region = df.groupby('Region').agg(
    {'Quantity Sold': 'mean', 'Profit Margin': 'mean'}).reset_index()
monthly_sales = df.groupby(df['Date'].str[:7])['Revenue'].sum().reset_index()

# Create the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Sales Performance Dashboard"),

    # Top 5 Products by Revenue
    html.Div([
        html.H3("Top 5 Products by Revenue"),
        dcc.Graph(
            id='top-products-bar',
            figure=px.bar(top_products, x='Product', y='Revenue',
                          title="Top 5 Products by Revenue")
        ),
    ]),

    # Total Revenue by Region
    html.Div([
        html.H3("Total Revenue by Region"),
        dcc.Graph(
            id='revenue-region-bar',
            figure=px.bar(revenue_by_region, x='Region',
                          y='Revenue', title="Total Revenue by Region")
        ),
    ]),

    # Average Sales and Profit Margin by Region
    html.Div([
        html.H3("Average Sales and Profit Margin by Region"),
        dcc.Graph(
            id='avg-sales-profit-region-bar',
            figure=px.bar(avg_sales_profit_by_region, x='Region', y=[
                          'Quantity Sold', 'Profit Margin'], title="Avg Sales & Profit Margin by Region")
        ),
    ]),

    # Monthly Sales Trend
    html.Div([
        html.H3("Monthly Sales Trend"),
        dcc.Graph(
            id='monthly-sales-line',
            figure=px.line(monthly_sales, x='Date', y='Revenue',
                           title="Monthly Sales Trend")
        ),
    ]),

    # Recommendations (Display as text)
    html.Div([
        html.H3("Actionable Recommendations"),
        html.P("""
            1. Focus marketing efforts on the top-performing products such as 'Apparel' and 'Electronics' to boost revenue.
            2. Invest in underperforming regions (e.g., South) to improve market share.
            3. Consider launching new promotions during months with lower sales (e.g., January) to increase revenue.
            4. Leverage customer demographics insights to target specific age groups for personalized marketing.
        """)
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
