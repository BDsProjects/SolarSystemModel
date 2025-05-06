# pages/extended.py
from dash import html, dcc, register_page
import solar_helpers_galactic as sh

register_page(__name__, path='/extended', name='Extended')
layout = html.Div([
    html.H2("Extended to Sedna"),
    dcc.Graph(figure=sh.build_fig(['Earth','Jupiter','Neptune','Pluto','Eris','Sedna', 'Orcus', 'Quaoar', 'Gonggong'], 1000, "Extended")),
])