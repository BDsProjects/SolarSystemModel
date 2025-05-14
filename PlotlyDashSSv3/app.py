# app.py
import dash
from dash import Dash, html, dcc, page_container

# Initialize Dash with explicit pages_folder
app = Dash(__name__, use_pages=True, pages_folder="pages")

app.layout = html.Div([
    html.H1("Solar System Explorer", style={'textAlign': 'center'}),
    # Auto-generated nav links
    html.Div([
        dcc.Link(page['name'], href=page['path'], style={'margin': '0 1em'})
        for page in dash.page_registry.values()
    ], style={'textAlign': 'center', 'padding': '10px'}),
    html.Hr(),
    # Render page content
    page_container
])

if __name__ == '__main__':
    app.run_server(debug=True)