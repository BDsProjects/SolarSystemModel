# pages/all.py
from dash import html, dcc, register_page
import solar_helpers_galactic as sh

register_page(__name__, path='/all', name='All')
layout = html.Div([
    html.H2("All Planets"),
    dcc.Graph(figure=sh.build_fig(['Mercury','Venus','Earth','Ceres','Mars','Jupiter','Saturn','Uranus','Neptune','Pluto','Haumea','Makemake','Eris', 'Sedna'], 1000, "All")),
])