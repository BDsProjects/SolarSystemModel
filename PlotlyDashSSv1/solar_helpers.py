# # solar_helpers.py
# import numpy as np
# import plotly.graph_objects as go

# # Shared data
# planet_colors = {
#     'Mercury': '#8c8680', 'Venus': '#e6c89c', 'Earth': '#4f71be', 'Mars': '#d1603d',
#     'Jupiter': '#e0ae6f', 'Saturn': '#c5ab6e', 'Uranus': '#9fc4e7', 'Neptune': '#4f71be',
#     'Ceres': '#8c8680', 'Pluto': '#ab9c8a', 'Eris': '#d9d9d9', 'Haumea': '#d9d9d9',
#     'Makemake': '#c49e6c', 'Sedna': '#bb5540'
# }

# orbital_params = {
#     'Mercury': [0.387, 0.2056, 7.005, 48.331, 29.124, 0.008],
#     'Venus':   [0.723, 0.0068, 3.39458, 76.680, 54.884, 0.02],
#     'Earth':   [1.0,   0.0167, 0.00005, -11.26064, 102.94719, 0.02],
#     'Mars':    [1.524, 0.0934, 1.850,    49.558,     286.502,   0.015],
#     'Jupiter': [5.2,   0.0489, 1.303,    100.464,    273.867,   0.045],
#     'Saturn':  [9.58,  0.0565, 2.485,    113.665,    339.392,   0.04],
#     'Uranus':  [19.22, 0.0457, 0.773,    74.006,     96.998,    0.035],
#     'Neptune':[30.05, 0.0113, 1.77,     131.783,    273.187,   0.035],
#     'Ceres':   [2.77,  0.0758, 10.593,   80.393,     73.597,    0.005],
#     'Pluto':   [39.48, 0.2488, 17.16,    110.299,    113.763,   0.01],
#     'Eris':    [67.8,  0.44068,44.04,    35.95,      151.639,   0.01],
#     'Haumea':  [43.13, 0.19126,28.19,    121.9,      239,       0.008],
#     'Makemake':[45.79, 0.159,  29,       79,         296,       0.008],
#     'Sedna':   [506,   0.8459, 11.93,    144.31,     311.46,    0.006]
# }

# # Helper: build Plotly traces for a planet list

# def make_traces(planet_list):
#     traces = []
#     # Sun sphere
#     u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
#     x_s = 0.05 * np.cos(u) * np.sin(v)
#     y_s = 0.05 * np.sin(u) * np.sin(v)
#     z_s = 0.05 * np.cos(v)
#     traces.append(go.Surface(x=x_s, y=y_s, z=z_s,
#                              colorscale=[[0,'yellow'],[1,'yellow']],
#                              opacity=0.7, showscale=False))
#     # Planets
#     for i, name in enumerate(planet_list):
#         a,e,inc,Ω,ω,rf = orbital_params[name]
#         inc,Ω,ω = np.radians([inc,Ω,ω])
#         θ = np.linspace(0,2*np.pi,500)
#         r = a*(1-e**2)/(1+e*np.cos(θ))
#         x = r*np.cos(θ); y = r*np.sin(θ); z = np.zeros_like(θ)
#         # rotate & incline
#         x1 = x*np.cos(ω)-y*np.sin(ω)
#         y1 = x*np.sin(ω)+y*np.cos(ω)
#         y2 = y1*np.cos(inc); z2 = y1*np.sin(inc)
#         x3 = x1*np.cos(Ω)-y2*np.sin(Ω)
#         y3 = x1*np.sin(Ω)+y2*np.cos(Ω)
#         # orbit line
#         traces.append(go.Scatter3d(x=x3, y=y3, z=z2, mode='lines',
#                                    line=dict(color=planet_colors[name], width=2), showlegend=False))
#         # current pos
#         t0 = np.radians((i*30)%360)
#         r0 = a*(1-e**2)/(1+e*np.cos(t0))
#         x0,y0 = r0*np.cos(t0), r0*np.sin(t0)
#         x0p = x0*np.cos(ω)-y0*np.sin(ω)
#         y0p = x0*np.sin(ω)+y0*np.cos(ω)
#         y0i = y0p*np.cos(inc); z0i = y0p*np.sin(inc)
#         x0r = x0p*np.cos(Ω)-y0i*np.sin(Ω)
#         y0r = x0p*np.sin(Ω)+y0i*np.cos(Ω)
#         traces.append(go.Scatter3d(x=[x0r], y=[y0r], z=[z0i],
#                                    mode='markers', marker=dict(size=8*rf, color=planet_colors[name]), name=name))
#     return traces

# # Helper: build a complete figure

# # def build_fig(subset, rng, title):
# #     fig = go.Figure(data=make_traces(subset))
# #     fig.update_layout(
# #         scene=dict(
# #             xaxis=dict(range=[-rng,rng], title='X (AU)'),
# #             yaxis=dict(range=[-rng,rng], title='Y (AU)'),
# #             zaxis=dict(range=[-rng,rng], title='Z (AU]'),
# #             aspectmode='data'
# #         ),
# #         margin=dict(l=0,r=0,t=40,b=0), title=title
# #     )
# #     return fig

# def build_fig(subset, rng, title):
#     fig = go.Figure(data=make_traces(subset))
#     fig.update_layout(
#         scene=dict(
#             xaxis=dict(range=[-rng,rng], title='X (AU)'),
#             yaxis=dict(range=[-rng,rng], title='Y (AU)'),
#             zaxis=dict(range=[-rng,rng], title='Z (AU)'),
#             aspectmode='cube',  # Change from 'data' to 'cube' for equal scaling but better use of space
#             camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # Adjust camera position
#         ),
#         margin=dict(l=10, r=10, t=50, b=10),  # Reduced margins
#         title=title,
#         height=1400,  # Set explicit height
#         width=1400,  # Allow width to be responsive
#         autosize=True,  # Enable autosize
#     )
#     return fig

# V2

# solar_helpers.py
import numpy as np
import plotly.graph_objects as go

# Shared data
planet_colors = {
    'Mercury': '#8c8680', 'Venus': '#e6c89c', 'Earth': '#4f71be', 'Mars': '#d1603d',
    'Jupiter': '#e0ae6f', 'Saturn': '#c5ab6e', 'Uranus': '#9fc4e7', 'Neptune': '#4f71be',
    'Ceres': '#8c8680', 'Pluto': '#ab9c8a', 'Eris': '#d9d9d9', 'Haumea': '#d9d9d9',
    'Makemake': '#c49e6c', 'Sedna': '#bb5540'
}

orbital_params = {
    'Mercury': [0.387, 0.2056, 7.005, 48.331, 29.124, 0.008],
    'Venus':   [0.723, 0.0068, 3.39458, 76.680, 54.884, 0.02],
    'Earth':   [1.0,   0.0167, 0.00005, -11.26064, 102.94719, 0.02],
    'Mars':    [1.524, 0.0934, 1.850,    49.558,     286.502,   0.015],
    'Jupiter': [5.2,   0.0489, 1.303,    100.464,    273.867,   0.045],
    'Saturn':  [9.58,  0.0565, 2.485,    113.665,    339.392,   0.04],
    'Uranus':  [19.22, 0.0457, 0.773,    74.006,     96.998,    0.035],
    'Neptune':[30.05, 0.0113, 1.77,     131.783,    273.187,   0.035],
    'Ceres':   [2.77,  0.0758, 10.593,   80.393,     73.597,    0.005],
    'Pluto':   [39.48, 0.2488, 17.16,    110.299,    113.763,   0.01],
    'Eris':    [67.8,  0.44068,44.04,    35.95,      151.639,   0.01],
    'Haumea':  [43.13, 0.19126,28.19,    121.9,      239,       0.008],
    'Makemake':[45.79, 0.159,  29,       79,         296,       0.008],
    'Sedna':   [506,   0.8459, 11.93,    144.31,     311.46,    0.006]
}

# Solar system region parameters (AU)
solar_regions = {
    'Asteroid Belt': {'inner_radius': 2.2, 'outer_radius': 3.2, 'color': 'gray', 'opacity': 0.2},
    'Kuiper Belt': {'inner_radius': 30, 'outer_radius': 50, 'color': 'lightblue', 'opacity': 0.15},
    'Heliopause': {'radius': 120, 'color': 'rgba(100,100,255,0.5)', 'opacity': 0.1},
    'Oort Cloud': {'inner_radius': 2000, 'outer_radius': 100000, 'color': 'rgba(200,200,255,0.3)', 'opacity': 0.05}
}

# Helper: create a spherical shell
def create_shell(inner_radius, outer_radius=None, color='gray', opacity=0.2, points=40):
    if outer_radius is None:  # Simple sphere
        u, v = np.mgrid[0:2*np.pi:points*1j, 0:np.pi:points//2*1j]
        x = inner_radius * np.cos(u) * np.sin(v)
        y = inner_radius * np.sin(u) * np.sin(v)
        z = inner_radius * np.cos(v)
        return go.Surface(x=x, y=y, z=z, colorscale=[[0, color], [1, color]], 
                         opacity=opacity, showscale=False)
    else:  # Shell with random points between inner and outer radius
        # Generate random points in a shell
        n_points = 5000
        phi = np.random.uniform(0, 2*np.pi, n_points)
        theta = np.arccos(np.random.uniform(-1, 1, n_points))
        r = np.random.uniform(inner_radius, outer_radius, n_points)
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        return go.Scatter3d(x=x, y=y, z=z, mode='markers',
                           marker=dict(size=1.5, color=color, opacity=opacity),
                           hoverinfo='none', showlegend=True)

# Helper: create a disk (for asteroid belt)
def create_disk(inner_radius, outer_radius, color='gray', opacity=0.2, points=1000):
    # Create scattered points to represent a disk
    r = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2, points))
    theta = np.random.uniform(0, 2*np.pi, points)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # Add some vertical dispersion for thickness
    z = np.random.normal(0, (outer_radius-inner_radius)*0.05, points)
    
    return go.Scatter3d(x=x, y=y, z=z, mode='markers',
                       marker=dict(size=1.5, color=color, opacity=opacity),
                       name='Asteroid Belt', hoverinfo='name')

# Helper: build Plotly traces for a planet list
def make_traces(planet_list, include_regions=True):
    traces = []
    # Sun sphere
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x_s = 0.05 * np.cos(u) * np.sin(v)
    y_s = 0.05 * np.sin(u) * np.sin(v)
    z_s = 0.05 * np.cos(v)
    traces.append(go.Surface(x=x_s, y=y_s, z=z_s,
                             colorscale=[[0,'yellow'],[1,'yellow']],
                             opacity=0.7, showscale=False))
    
    # Add solar system regions if requested
    if include_regions:
        # Asteroid Belt (disk representation)
        ab_params = solar_regions['Asteroid Belt']
        traces.append(create_disk(ab_params['inner_radius'], ab_params['outer_radius'], 
                               ab_params['color'], ab_params['opacity']))
        
        # Kuiper Belt (shell with scattered points)
        kb_params = solar_regions['Kuiper Belt']
        kb_trace = create_shell(kb_params['inner_radius'], kb_params['outer_radius'], 
                              kb_params['color'], kb_params['opacity'])
        kb_trace.name = 'Kuiper Belt'
        traces.append(kb_trace)
        
        # Heliopause (spherical boundary)
        hp_params = solar_regions['Heliopause']
        hp_trace = create_shell(hp_params['radius'], None, hp_params['color'], hp_params['opacity'])
        hp_trace.name = 'Heliopause'
        traces.append(hp_trace)
        
        # Oort Cloud is typically not shown in the same scale view as planets
        # as it's thousands of AU away, but we can add it as an option
    
    # Planets
    for i, name in enumerate(planet_list):
        a,e,inc,Ω,ω,rf = orbital_params[name]
        inc,Ω,ω = np.radians([inc,Ω,ω])
        θ = np.linspace(0,2*np.pi,500)
        r = a*(1-e**2)/(1+e*np.cos(θ))
        x = r*np.cos(θ); y = r*np.sin(θ); z = np.zeros_like(θ)
        # rotate & incline
        x1 = x*np.cos(ω)-y*np.sin(ω)
        y1 = x*np.sin(ω)+y*np.cos(ω)
        y2 = y1*np.cos(inc); z2 = y1*np.sin(inc)
        x3 = x1*np.cos(Ω)-y2*np.sin(Ω)
        y3 = x1*np.sin(Ω)+y2*np.cos(Ω)
        # orbit line
        traces.append(go.Scatter3d(x=x3, y=y3, z=z2, mode='lines',
                                   line=dict(color=planet_colors[name], width=2), showlegend=False))
        # current pos
        t0 = np.radians((i*30)%360)
        r0 = a*(1-e**2)/(1+e*np.cos(t0))
        x0,y0 = r0*np.cos(t0), r0*np.sin(t0)
        x0p = x0*np.cos(ω)-y0*np.sin(ω)
        y0p = x0*np.sin(ω)+y0*np.cos(ω)
        y0i = y0p*np.cos(inc); z0i = y0p*np.sin(inc)
        x0r = x0p*np.cos(Ω)-y0i*np.sin(Ω)
        y0r = x0p*np.sin(Ω)+y0i*np.cos(Ω)
        traces.append(go.Scatter3d(x=[x0r], y=[y0r], z=[z0i],
                                   mode='markers', marker=dict(size=8*rf, color=planet_colors[name]), name=name))
    return traces

# Helper: build a complete figure
def build_fig(subset, rng, title, include_regions=True, include_oort=False):
    traces = make_traces(subset, include_regions)
    
    # Add Oort Cloud separately if requested (only for very large scale views)
    if include_oort:
        oc_params = solar_regions['Oort Cloud']
        oc_trace = create_shell(oc_params['inner_radius'], oc_params['outer_radius'], 
                              oc_params['color'], oc_params['opacity'])
        oc_trace.name = 'Oort Cloud'
        traces.append(oc_trace)
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-rng,rng], title='X (AU)'),
            yaxis=dict(range=[-rng,rng], title='Y (AU)'),
            zaxis=dict(range=[-rng,rng], title='Z (AU)'),
            aspectmode='cube',  # Change from 'data' to 'cube' for equal scaling but better use of space
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # Adjust camera position
        ),
        margin=dict(l=10, r=10, t=50, b=10),  # Reduced margins
        title=title,
        height=700,  # Set explicit height
        width=700,  # Set explicit width
        autosize=True,  # Enable autosize
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    return fig