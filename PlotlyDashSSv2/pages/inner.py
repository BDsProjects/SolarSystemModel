# pages/inner.py
from dash import html, dcc, register_page
import solar_helpers as sh

register_page(__name__, path='/inner', name='Inner')
layout = html.Div([
    html.H2("Inner Solar System"),
    dcc.Graph(figure=sh.build_fig(['Mercury','Venus','Earth','Mars','Ceres'], 3, "Inner")),
])