from interactive_dashboard import app, df  # Import the app and dataset
from dash import dcc, html, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.io as pio

# ===== Add New Layout Components =====
app.layout.children.extend([
    # Key Metrics Cards
    html.Div([
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4("Total Revenue"),
                    html.H2(f"${df['Revenue'].sum():,.2f}")
                ])
            ], color="primary", inverse=True)),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4("Total Quantity Sold"),
                    html.H2(f"{df['Quantity Sold'].sum()} units")
                ])
            ], color="success", inverse=True)),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4("Average Profit Margin"),
                    html.H2(f"{df['Profit Margin'].mean():.2f}%")
                ])
            ], color="info", inverse=True)),
        ])
    ], style={"margin-bottom": "20px"}),

    # Drill-Down Graph Section
    html.Div([
        dcc.Graph(id='drill-down-graph'),
        html.P("Click on a bar in the Revenue graph to see daily details.")
    ]),

    # Export Graph as Image
    html.Div([
        html.Button("Download Revenue Graph", id="download-graph", n_clicks=0),
    ], style={"margin-top": "20px"})
])

# ===== Drill-Down Callback =====
@app.callback(
    Output('drill-down-graph', 'figure'),
    [Input('revenue-trend', 'clickData')]
)
def update_drill_down(clickData):
    if clickData:
        selected_date = clickData['points'][0]['x']  # Get clicked date
        filtered_df = df[df['Date'] == selected_date]  # Filter data
        fig = px.bar(filtered_df, x='Product', y='Revenue', 
                     title=f"Daily Revenue Details for {selected_date}")
    else:
        fig = px.bar(title="Click on a bar in the Revenue graph.")
    return fig

# ===== Export Graph as Image Callback =====
@app.callback(
    Output('download-dataframe-csv', 'data'),
    [Input('download-graph', 'n_clicks')],
    [State('revenue-trend', 'figure')]
)
def export_graph_as_image(n_clicks, figure):
    if n_clicks > 0:
        pio.write_image(figure, "revenue_graph.png")  # Save graph as PNG
        return "Graph saved as PNG"
