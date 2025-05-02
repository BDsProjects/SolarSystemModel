# pages/outer.py
from dash import html, dcc, register_page
import solar_helpers_galactic as sh

register_page(__name__, path='/outer', name='Outer')
layout = html.Div([
    html.H2("Outer & Dwarfs"),
    dcc.Graph(figure=sh.build_fig(['Jupiter','Saturn','Uranus','Neptune','Pluto','Haumea','Makemake','Eris'], 70, "Outer & Dwarfs")),
])