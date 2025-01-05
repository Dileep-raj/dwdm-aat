import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px

# Load your dataset
df = pd.read_csv('sales_data_with_clusters.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define a sample chart (e.g., Total Revenue by Region)
fig = px.bar(df, x='Region', y='Revenue', title='Total Revenue by Region')

# Define the layout with better structure and styling
app.layout = html.Div([
    html.Header([
        html.H1("Sales Performance Dashboard", style={'text-align': 'center', 'color': '#2C3E50', 'font-size': '36px'}),
        html.P("An interactive dashboard showcasing sales trends, revenue, and performance across regions.", style={'text-align': 'center', 'font-size': '20px'})
    ], style={'background-color': '#F1F3F4', 'padding': '20px'}),

    html.Div([
        html.Div([
            html.Label("Select Region:", style={'font-size': '18px'}),
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': region, 'value': region} for region in df['Region'].unique()],
                placeholder="Select a region",
                style={'width': '50%', 'padding': '10px', 'font-size': '16px'}
            )
        ], style={'padding': '20px', 'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(
                id='revenue-graph',
                figure=fig
            )
        ], style={'padding': '20px', 'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'})
    ], style={'display': 'flex', 'justify-content': 'space-between'}),

    # Add more charts or components here
    html.Div([
        html.H2("Revenue Overview", style={'text-align': 'center'}),
        dcc.Graph(
            id='total-revenue-chart',
            figure=fig  # Use any other chart here
        )
    ], style={'padding': '20px'})
])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
