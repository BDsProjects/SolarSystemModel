# pages/test.py
from dash import html, dcc, register_page
import solar_helpers as sh

register_page(__name__, path='/test', name='Test')
layout = html.Div([
    html.H2("Inner Solar System"),
    dcc.Graph(figure=sh.build_fig(['Earth'], 3, "Test")),
])