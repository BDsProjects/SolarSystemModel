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

# from dash import html, dcc, register_page
# import solar_helpers_galactic as sh

# register_page(__name__, path='/all_and_regions', name='All with Regions')

# layout = html.Div([
#     html.H2("All Planets and Regions"),
#     dcc.Graph(
#         figure=sh.build_fig(
#             ['Mercury', 'Venus', 'Earth', 'Ceres', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto', 'Haumea', 'Makemake', 'Eris', 'Sedna'], 
#             5000, 
#             "All Planets with Solar System Regions",
#             include_regions=True,
#             include_ellipsoid=True,  # Enable the ellipsoid
#             ellipsoid_params={
#                 'inner_edge': -120,
#                 'length': 900,
#                 'width_factor': 0.75,
#                 'height_factor': 0.5,
#                 'color': 'rgba(255,100,100,0.3)', 
#                 'opacity': 0.2,
#                 'name': 'Extended Region'
#             }
#         ),
#         style={'height': '85vh'}
#     ),
#     html.Div([
#         html.P("This visualization shows all planets along with the asteroid belt, Kuiper belt, heliopause, and a special extended region (red ellipsoid) that starts at 120 AU and extends 300 AU outward.")
#     ], className="mt-2")
# ])

from dash import html, dcc, register_page
import solar_helpers_galactic as sh

register_page(__name__, path='/all_and_regions', name='All with Regions')

# Create a figure with all planets and regions including termination shock
# def build_complete_figure():
#     # Create the basic figure first
#     fig = sh.build_fig(
#         ['Mercury', 'Venus', 'Earth', 'Ceres', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto', 'Haumea', 'Makemake', 'Eris', 'Sedna'], 
#         5000, 
#         "All Planets with Solar System Regions",
#         include_regions=True,
#         include_ellipsoid=False,  # Enable the ellipsoid
#         ellipsoid_params={
#             'inner_edge': -120,
#             'length': 900,
#             'width_factor': 0.75,
#             'height_factor': 0.5,
#             'color': 'rgba(255,100,100,0.3)', 
#             'opacity': 0.2,
#             'name': 'Extended Region'
#         }
#     )
    
#     # Add termination shock
#     ts_trace = sh.create_termination_shock(
#         inner_radius=80,
#         outer_radius=100,
#         tailward_offset=80,
#         north_distance=20,
#         port_side=15,
#         color='rgba(255,100,100,0.3)',
#         opacity=0.3
#     )
#     fig.add_trace(ts_trace)
    
#     # Add heliotail
#     ht_trace = sh.create_heliotail(
#         base_radius=100,
#         length=800,
#         width_factor=0.6,
#         height_factor=0.5,
#         color='rgba(200,100,100,0.2)',
#         opacity=0.15
#     )
#     fig.add_trace(ht_trace)
    
#     # Add bow shock
#     bs_trace = sh.create_bow_shock(
#         standoff_distance=130,
#         radius=180,
#         thickness=15,
#         color='rgba(255,150,100,0.25)',
#         opacity=0.2
#     )
#     fig.add_trace(bs_trace)
    
#     return fig
# Updated build_complete_figure function that uses the new bow shock
from solar_helpers_galactic import create_bow_shock_ellipsoid

def build_complete_figure():
    # Create the basic figure first
    fig = sh.build_fig(
        ['Mercury', 'Venus', 'Earth', 'Ceres', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto', 'Haumea', 'Makemake', 'Eris', 'Sedna'], 
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
    )
    
    # Add termination shock
    ts_trace = sh.create_termination_shock(
        inner_radius=80,
        outer_radius=100,
        tailward_offset=80,
        north_distance=20,
        port_side=15,
        color='rgba(255,100,100,0.3)',
        opacity=0.3
    )
    fig.add_trace(ts_trace)
    
    # Add heliotail
    ht_trace = sh.create_heliotail(
        base_radius=100,
        length=800,
        width_factor=0.6,
        height_factor=0.5,
        color='rgba(200,100,100,0.2)',
        opacity=0.15
    )
    fig.add_trace(ht_trace)
    
    # Add bow shock as ellipsoid
    bs_trace = create_bow_shock_ellipsoid(
    standoff_distance=130,
    base_radius=180,
    length=1500,
    width_factor=0.8,
    height_factor=0.5,
    color='rgba(255,150,100,0.25)',
    opacity=0.2
    )
    fig.add_trace(bs_trace)
    
    return fig

layout = html.Div([
    html.H2("All Planets and Regions"),
    dcc.Graph(
        figure=build_complete_figure(),
        style={'height': '85vh'}
    ),
    html.Div([
        html.P("This visualization shows all planets along with the asteroid belt, Kuiper belt, heliopause, and the heliosphere structure including the termination shock (inner red boundary), heliotail (extending in the anti-sun direction), and bow shock (outer boundary).")
    ], className="mt-2")
])