# pages/main.py
from dash import html, dcc, register_page
import solar_helpers as sh

register_page(__name__, path='/main', name='Main')
layout = html.Div([
    html.H2("Main Planets"),
    dcc.Graph(figure=sh.build_fig(['Earth','Mars','Jupiter','Saturn','Uranus','Neptune'], 35, "Main")),
])