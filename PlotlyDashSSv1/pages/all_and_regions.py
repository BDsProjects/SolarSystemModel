# # pages/all_and_regions.py

# from dash import html, dcc, register_page
# import solar_helpers as sh

# register_page(__name__, path='/all_and_regions', name='All with Regions')
# layout = html.Div([
#     html.H2("All Planets and Regions"),
#     dcc.Graph(figure=sh.build_fig(['Mercury','Venus','Earth','Ceres','Mars','Jupiter','Saturn','Uranus','Neptune','Pluto','Haumea','Makemake','Eris', 'Sedna'], 1000, "All")),
# ])


# pages/all_and_regions.py

# from dash import html, dcc, register_page
# import solar_helpers as sh

# register_page(__name__, path='/all_and_regions', name='All with Regions')

# layout = html.Div([
#     html.H2("All Planets and Regions"),
#     dcc.Graph(
#         figure=sh.build_fig(
#             ['Mercury', 'Venus', 'Earth', 'Ceres', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto', 'Haumea', 'Makemake', 'Eris', 'Sedna'], 
#             5000, 
#             "All Planets with Solar System Regions",
#             include_regions=True
#         ),
#         style={'height': '85vh'}
#     ),
#     html.Div([
#         html.P("This visualization shows all planets along with the asteroid belt, Kuiper belt, and heliopause.")
#     ], className="mt-2")
# ])

from dash import html, dcc, register_page
import solar_helpers as sh

register_page(__name__, path='/all_and_regions', name='All with Regions')

layout = html.Div([
    html.H2("All Planets and Regions"),
    dcc.Graph(
        figure=sh.build_fig(
            ['Mercury', 'Venus', 'Earth', 'Ceres', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto', 'Haumea', 'Makemake', 'Eris', 'Sedna', 'Orcus', 'Quaoar', 'Gonggong'], 
            5000, 
            "All Planets with Solar System Regions",
            include_regions=True,
            include_ellipsoid=True,  # Enable the ellipsoid
            ellipsoid_params={
                'inner_edge': -120,
                'length': 900,
                'width_factor': 0.75,
                'height_factor': 0.5,
                'color': 'rgba(255,100,100,0.3)', 
                'opacity': 0.2,
                'name': 'Extended Region'
            }
        ),
        style={'height': '85vh'}
    ),
    html.Div([
        html.P("This visualization shows all planets along with the asteroid belt, Kuiper belt, heliopause, and a special extended region (red ellipsoid) that starts at 120 AU and extends 300 AU outward.")
    ], className="mt-2")
])