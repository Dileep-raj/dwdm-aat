import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import io
import base64
import numpy as np
from export_functions import export_to_csv




# Load your data (make sure you have the correct path to your dataset)
df = pd.read_csv('sales_data_with_clusters.csv')

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define app layout
app.layout = html.Div([
    html.H1("Sales Performance Dashboard", style={"text-align": "center"}),

    # Dropdown for region selection
    html.Div([
        dcc.Dropdown(
            id='region-dropdown',
            options=[{'label': region, 'value': region} for region in df['Region'].unique()],
            value='East',  # Default value
            placeholder="Select Region"
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    # Dropdown for product category selection
    html.Div([
        dcc.Dropdown(
            id='product-dropdown',
            options=[{'label': product, 'value': product} for product in df['Product'].unique()],
            value='Electronics',  # Default value
            placeholder="Select Product Category"
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    # Date picker range for selecting a date range
    html.Div([
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=df['Date'].min().date(),
            end_date=df['Date'].max().date(),
            display_format='YYYY-MM-DD'
        ),
    ], style={'width': '48%', 'display': 'inline-block', 'padding-top': '10px'}),

    # Dropdown for selecting graph type
    html.Div([
        dcc.Dropdown(
            id='graph-type-dropdown',
            options=[
                {'label': 'Line Graph', 'value': 'line'},
                {'label': 'Bar Graph', 'value': 'bar'},
                {'label': 'Scatter Plot', 'value': 'scatter'}
            ],
            value='line',  # Default value
            placeholder="Select Graph Type"
        ),
    ], style={'width': '48%', 'display': 'inline-block', 'padding-top': '10px'}),

    # Graph for showing sales trends (Revenue)
    dcc.Graph(id='revenue-trend'),

    # Graph for showing quantity sold trends
    dcc.Graph(id='quantity-sold-trend'),

    # Graph for showing profit margin trends
    dcc.Graph(id='profit-margin-trend'),

    # Table for showing top 5 products
    dash_table.DataTable(id='top-products-table'),

    # Export button for CSV
    html.Div([
        html.Button("Export Data", id="export-csv", n_clicks=0),
        dcc.Download(id="download-dataframe-csv")
    ]),

    # Insights Section
    html.Div(id="insights")
])





# Callbacks for dynamic updates

# Update the revenue trend graph based on filters and graph type
@app.callback(
    Output('revenue-trend', 'figure'),
    [Input('region-dropdown', 'value'),
     Input('product-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('graph-type-dropdown', 'value')]
)
def update_revenue_trend(selected_region, selected_product, start_date, end_date, graph_type):
    filtered_df = df[
        (df['Region'] == selected_region) &
        (df['Product'] == selected_product) &
        (df['Date'] >= start_date) &
        (df['Date'] <= end_date)
    ]
    revenue_trend = filtered_df.groupby('Date')['Revenue'].sum().reset_index()

    # Create the graph based on the selected type
    if graph_type == 'line':
        fig = px.line(revenue_trend, x='Date', y='Revenue', title=f"Revenue Trend for {selected_product} in {selected_region}")
    elif graph_type == 'bar':
        fig = px.bar(revenue_trend, x='Date', y='Revenue', title=f"Revenue Trend for {selected_product} in {selected_region}")
    elif graph_type == 'scatter':
        fig = px.scatter(revenue_trend, x='Date', y='Revenue', title=f"Revenue Trend for {selected_product} in {selected_region}")

    return fig


# Update the quantity sold trend graph
@app.callback(
    Output('quantity-sold-trend', 'figure'),
    [Input('region-dropdown', 'value'),
     Input('product-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('graph-type-dropdown', 'value')]
)
def update_quantity_sold_trend(selected_region, selected_product, start_date, end_date, graph_type):
    filtered_df = df[
        (df['Region'] == selected_region) &
        (df['Product'] == selected_product) &
        (df['Date'] >= start_date) &
        (df['Date'] <= end_date)
    ]
    quantity_sold_trend = filtered_df.groupby('Date')['Quantity Sold'].sum().reset_index()

    # Create the graph based on the selected type
    if graph_type == 'line':
        fig = px.line(quantity_sold_trend, x='Date', y='Quantity Sold', title=f"Quantity Sold Trend for {selected_product} in {selected_region}")
    elif graph_type == 'bar':
        fig = px.bar(quantity_sold_trend, x='Date', y='Quantity Sold', title=f"Quantity Sold Trend for {selected_product} in {selected_region}")
    elif graph_type == 'scatter':
        fig = px.scatter(quantity_sold_trend, x='Date', y='Quantity Sold', title=f"Quantity Sold Trend for {selected_product} in {selected_region}")

    return fig


# Update the profit margin trend graph
@app.callback(
    Output('profit-margin-trend', 'figure'),
    [Input('region-dropdown', 'value'),
     Input('product-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('graph-type-dropdown', 'value')]
)
def update_profit_margin_trend(selected_region, selected_product, start_date, end_date, graph_type):
    filtered_df = df[
        (df['Region'] == selected_region) &
        (df['Product'] == selected_product) &
        (df['Date'] >= start_date) &
        (df['Date'] <= end_date)
    ]
    profit_margin_trend = filtered_df.groupby('Date')['Profit Margin'].mean().reset_index()

    # Create the graph based on the selected type
    if graph_type == 'line':
        fig = px.line(profit_margin_trend, x='Date', y='Profit Margin', title=f"Profit Margin Trend for {selected_product} in {selected_region}")
    elif graph_type == 'bar':
        fig = px.bar(profit_margin_trend, x='Date', y='Profit Margin', title=f"Profit Margin Trend for {selected_product} in {selected_region}")
    elif graph_type == 'scatter':
        fig = px.scatter(profit_margin_trend, x='Date', y='Profit Margin', title=f"Profit Margin Trend for {selected_product} in {selected_region}")

    return fig


# Update the top 5 products table
@app.callback(
    Output('top-products-table', 'data'),
    [Input('region-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_top_products_table(selected_region, start_date, end_date):
    filtered_df = df[
        (df['Region'] == selected_region) &
        (df['Date'] >= start_date) &
        (df['Date'] <= end_date)
    ]
    top_products = filtered_df.groupby('Product').agg({'Revenue': 'sum', 'Quantity Sold': 'sum', 'Profit Margin': 'mean'}).reset_index()
    top_products = top_products.sort_values('Revenue', ascending=False).head(5)
    return top_products.to_dict('records')


# Generate insights based on the selection
@app.callback(
    Output('insights', 'children'),
    [Input('region-dropdown', 'value'),
     Input('product-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_insights(selected_region, selected_product, start_date, end_date):
    filtered_df = df[
        (df['Region'] == selected_region) &
        (df['Product'] == selected_product) &
        (df['Date'] >= start_date) &
        (df['Date'] <= end_date)
    ]

    # Total Revenue
    total_revenue = filtered_df['Revenue'].sum()

    # Total Quantity Sold
    total_quantity_sold = filtered_df['Quantity Sold'].sum()

    # Profit Margin
    profit_margin = filtered_df['Profit Margin'].mean()

    insights = [
        html.H3(f"Insights for {selected_product} in {selected_region}"),
        html.P(f"Total Revenue: ${total_revenue:,.2f}"),
        html.P(f"Total Quantity Sold: {total_quantity_sold} units"),
        html.P(f"Average Profit Margin: {profit_margin:.2f}%")
    ]

    return insights


# Export data as CSV
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("export-csv", "n_clicks"),
    State('region-dropdown', 'value'),
    State('product-dropdown', 'value'),
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date'),
)
def export_csv(n_clicks, selected_region, selected_product, start_date, end_date):
    if n_clicks > 0:
        filtered_df = df[
            (df['Region'] == selected_region) &
            (df['Product'] == selected_product) &
            (df['Date'] >= start_date) &
            (df['Date'] <= end_date)
        ]
        return dcc.send_data_frame(filtered_df.to_csv, "sales_data.csv")
    return None



if __name__ == '__main__':
    app.run_server(debug=True)

