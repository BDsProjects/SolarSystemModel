# # # solar_helpers.py
# # import numpy as np
# # import plotly.graph_objects as go

# # # Shared data
# # planet_colors = {
# #     'Mercury': '#8c8680', 'Venus': '#e6c89c', 'Earth': '#4f71be', 'Mars': '#d1603d',
# #     'Jupiter': '#e0ae6f', 'Saturn': '#c5ab6e', 'Uranus': '#9fc4e7', 'Neptune': '#4f71be',
# #     'Ceres': '#8c8680', 'Pluto': '#ab9c8a', 'Eris': '#d9d9d9', 'Haumea': '#d9d9d9',
# #     'Makemake': '#c49e6c', 'Sedna': '#bb5540'
# # }

# # orbital_params = {
# #     'Mercury': [0.387, 0.2056, 7.005, 48.331, 29.124, 0.008],
# #     'Venus':   [0.723, 0.0068, 3.39458, 76.680, 54.884, 0.02],
# #     'Earth':   [1.0,   0.0167, 0.00005, -11.26064, 102.94719, 0.02],
# #     'Mars':    [1.524, 0.0934, 1.850,    49.558,     286.502,   0.015],
# #     'Jupiter': [5.2,   0.0489, 1.303,    100.464,    273.867,   0.045],
# #     'Saturn':  [9.58,  0.0565, 2.485,    113.665,    339.392,   0.04],
# #     'Uranus':  [19.22, 0.0457, 0.773,    74.006,     96.998,    0.035],
# #     'Neptune':[30.05, 0.0113, 1.77,     131.783,    273.187,   0.035],
# #     'Ceres':   [2.77,  0.0758, 10.593,   80.393,     73.597,    0.005],
# #     'Pluto':   [39.48, 0.2488, 17.16,    110.299,    113.763,   0.01],
# #     'Eris':    [67.8,  0.44068,44.04,    35.95,      151.639,   0.01],
# #     'Haumea':  [43.13, 0.19126,28.19,    121.9,      239,       0.008],
# #     'Makemake':[45.79, 0.159,  29,       79,         296,       0.008],
# #     'Sedna':   [506,   0.8459, 11.93,    144.31,     311.46,    0.006]
# # }

# # # Helper: build Plotly traces for a planet list

# # def make_traces(planet_list):
# #     traces = []
# #     # Sun sphere
# #     u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
# #     x_s = 0.05 * np.cos(u) * np.sin(v)
# #     y_s = 0.05 * np.sin(u) * np.sin(v)
# #     z_s = 0.05 * np.cos(v)
# #     traces.append(go.Surface(x=x_s, y=y_s, z=z_s,
# #                              colorscale=[[0,'yellow'],[1,'yellow']],
# #                              opacity=0.7, showscale=False))
# #     # Planets
# #     for i, name in enumerate(planet_list):
# #         a,e,inc,Ω,ω,rf = orbital_params[name]
# #         inc,Ω,ω = np.radians([inc,Ω,ω])
# #         θ = np.linspace(0,2*np.pi,500)
# #         r = a*(1-e**2)/(1+e*np.cos(θ))
# #         x = r*np.cos(θ); y = r*np.sin(θ); z = np.zeros_like(θ)
# #         # rotate & incline
# #         x1 = x*np.cos(ω)-y*np.sin(ω)
# #         y1 = x*np.sin(ω)+y*np.cos(ω)
# #         y2 = y1*np.cos(inc); z2 = y1*np.sin(inc)
# #         x3 = x1*np.cos(Ω)-y2*np.sin(Ω)
# #         y3 = x1*np.sin(Ω)+y2*np.cos(Ω)
# #         # orbit line
# #         traces.append(go.Scatter3d(x=x3, y=y3, z=z2, mode='lines',
# #                                    line=dict(color=planet_colors[name], width=2), showlegend=False))
# #         # current pos
# #         t0 = np.radians((i*30)%360)
# #         r0 = a*(1-e**2)/(1+e*np.cos(t0))
# #         x0,y0 = r0*np.cos(t0), r0*np.sin(t0)
# #         x0p = x0*np.cos(ω)-y0*np.sin(ω)
# #         y0p = x0*np.sin(ω)+y0*np.cos(ω)
# #         y0i = y0p*np.cos(inc); z0i = y0p*np.sin(inc)
# #         x0r = x0p*np.cos(Ω)-y0i*np.sin(Ω)
# #         y0r = x0p*np.sin(Ω)+y0i*np.cos(Ω)
# #         traces.append(go.Scatter3d(x=[x0r], y=[y0r], z=[z0i],
# #                                    mode='markers', marker=dict(size=8*rf, color=planet_colors[name]), name=name))
# #     return traces

# # # Helper: build a complete figure

# # # def build_fig(subset, rng, title):
# # #     fig = go.Figure(data=make_traces(subset))
# # #     fig.update_layout(
# # #         scene=dict(
# # #             xaxis=dict(range=[-rng,rng], title='X (AU)'),
# # #             yaxis=dict(range=[-rng,rng], title='Y (AU)'),
# # #             zaxis=dict(range=[-rng,rng], title='Z (AU]'),
# # #             aspectmode='data'
# # #         ),
# # #         margin=dict(l=0,r=0,t=40,b=0), title=title
# # #     )
# # #     return fig

# # def build_fig(subset, rng, title):
# #     fig = go.Figure(data=make_traces(subset))
# #     fig.update_layout(
# #         scene=dict(
# #             xaxis=dict(range=[-rng,rng], title='X (AU)'),
# #             yaxis=dict(range=[-rng,rng], title='Y (AU)'),
# #             zaxis=dict(range=[-rng,rng], title='Z (AU)'),
# #             aspectmode='cube',  # Change from 'data' to 'cube' for equal scaling but better use of space
# #             camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # Adjust camera position
# #         ),
# #         margin=dict(l=10, r=10, t=50, b=10),  # Reduced margins
# #         title=title,
# #         height=1400,  # Set explicit height
# #         width=1400,  # Allow width to be responsive
# #         autosize=True,  # Enable autosize
# #     )
# #     return fig

# # V2

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

# # Solar system region parameters (AU)
# solar_regions = {
#     'Asteroid Belt': {'inner_radius': 2.2, 'outer_radius': 3.2, 'color': 'gray', 'opacity': 0.2},
#     'Kuiper Belt': {'inner_radius': 30, 'outer_radius': 50, 'color': 'lightblue', 'opacity': 0.15},
#     'Heliopause': {'radius': 120, 'color': 'rgba(100,100,255,0.5)', 'opacity': 0.1},
#     'Oort Cloud': {'inner_radius': 2000, 'outer_radius': 100000, 'color': 'rgba(200,200,255,0.3)', 'opacity': 0.05}
# }

# # Helper: create a spherical shell
# def create_shell(inner_radius, outer_radius=None, color='gray', opacity=0.2, points=40):
#     if outer_radius is None:  # Simple sphere
#         u, v = np.mgrid[0:2*np.pi:points*1j, 0:np.pi:points//2*1j]
#         x = inner_radius * np.cos(u) * np.sin(v)
#         y = inner_radius * np.sin(u) * np.sin(v)
#         z = inner_radius * np.cos(v)
#         return go.Surface(x=x, y=y, z=z, colorscale=[[0, color], [1, color]], 
#                          opacity=opacity, showscale=False)
#     else:  # Shell with random points between inner and outer radius
#         # Generate random points in a shell
#         n_points = 5000
#         phi = np.random.uniform(0, 2*np.pi, n_points)
#         theta = np.arccos(np.random.uniform(-1, 1, n_points))
#         r = np.random.uniform(inner_radius, outer_radius, n_points)
        
#         x = r * np.sin(theta) * np.cos(phi)
#         y = r * np.sin(theta) * np.sin(phi)
#         z = r * np.cos(theta)
        
#         return go.Scatter3d(x=x, y=y, z=z, mode='markers',
#                            marker=dict(size=1.5, color=color, opacity=opacity),
#                            hoverinfo='none', showlegend=True)

# # Helper: create a disk (for asteroid belt)
# def create_disk(inner_radius, outer_radius, color='gray', opacity=0.2, points=1000):
#     # Create scattered points to represent a disk
#     r = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2, points))
#     theta = np.random.uniform(0, 2*np.pi, points)
    
#     x = r * np.cos(theta)
#     y = r * np.sin(theta)
#     # Add some vertical dispersion for thickness
#     z = np.random.normal(0, (outer_radius-inner_radius)*0.05, points)
    
#     return go.Scatter3d(x=x, y=y, z=z, mode='markers',
#                        marker=dict(size=1.5, color=color, opacity=opacity),
#                        name='Asteroid Belt', hoverinfo='name')

# # Helper: build Plotly traces for a planet list
# def make_traces(planet_list, include_regions=True):
#     traces = []
#     # Sun sphere
#     u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
#     x_s = 0.05 * np.cos(u) * np.sin(v)
#     y_s = 0.05 * np.sin(u) * np.sin(v)
#     z_s = 0.05 * np.cos(v)
#     traces.append(go.Surface(x=x_s, y=y_s, z=z_s,
#                              colorscale=[[0,'yellow'],[1,'yellow']],
#                              opacity=0.7, showscale=False))
    
#     # Add solar system regions if requested
#     if include_regions:
#         # Asteroid Belt (disk representation)
#         ab_params = solar_regions['Asteroid Belt']
#         traces.append(create_disk(ab_params['inner_radius'], ab_params['outer_radius'], 
#                                ab_params['color'], ab_params['opacity']))
        
#         # Kuiper Belt (shell with scattered points)
#         kb_params = solar_regions['Kuiper Belt']
#         kb_trace = create_shell(kb_params['inner_radius'], kb_params['outer_radius'], 
#                               kb_params['color'], kb_params['opacity'])
#         kb_trace.name = 'Kuiper Belt'
#         traces.append(kb_trace)
        
#         # Heliopause (spherical boundary)
#         hp_params = solar_regions['Heliopause']
#         hp_trace = create_shell(hp_params['radius'], None, hp_params['color'], hp_params['opacity'])
#         hp_trace.name = 'Heliopause'
#         traces.append(hp_trace)
        
#         # Oort Cloud is typically not shown in the same scale view as planets
#         # as it's thousands of AU away, but we can add it as an option
    
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
# def build_fig(subset, rng, title, include_regions=True, include_oort=False):
#     traces = make_traces(subset, include_regions)
    
#     # Add Oort Cloud separately if requested (only for very large scale views)
#     if include_oort:
#         oc_params = solar_regions['Oort Cloud']
#         oc_trace = create_shell(oc_params['inner_radius'], oc_params['outer_radius'], 
#                               oc_params['color'], oc_params['opacity'])
#         oc_trace.name = 'Oort Cloud'
#         traces.append(oc_trace)
    
#     fig = go.Figure(data=traces)
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
#         height=700,  # Set explicit height
#         width=700,  # Set explicit width
#         autosize=True,  # Enable autosize
#         legend=dict(
#             yanchor="top",
#             y=0.99,
#             xanchor="left",
#             x=0.01
#         )
#     )
#     return fig

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

# # Relative planetary radii (Earth = 1)
# planet_relative_sizes = {
#     'Mercury': 0.383, 'Venus': 0.950, 'Earth': 1.0, 'Mars': 0.532,
#     'Jupiter': 11.21, 'Saturn': 9.45, 'Uranus': 4.01, 'Neptune': 3.88,
#     'Ceres': 0.074, 'Pluto': 0.186, 'Eris': 0.183, 'Haumea': 0.16,
#     'Makemake': 0.18, 'Sedna': 0.10
# }

# # Scale factor to make planets more visible while maintaining relative proportions
# size_scale_factor = 0.15

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

# # Solar system region parameters (AU)
# solar_regions = {
#     'Asteroid Belt': {'inner_radius': 2.2, 'outer_radius': 3.2, 'color': 'gray', 'opacity': 0.2},
#     'Kuiper Belt': {'inner_radius': 30, 'outer_radius': 50, 'color': 'lightblue', 'opacity': 0.15},
#     'Heliopause': {'radius': 120, 'color': 'rgba(100,100,255,0.5)', 'opacity': 0.1},
#     'Oort Cloud': {'inner_radius': 2000, 'outer_radius': 100000, 'color': 'rgba(200,200,255,0.3)', 'opacity': 0.05}
# }

# # Helper: create a spherical shell
# def create_shell(inner_radius, outer_radius=None, color='gray', opacity=0.2, points=40):
#     if outer_radius is None:  # Simple sphere
#         u, v = np.mgrid[0:2*np.pi:points*1j, 0:np.pi:points//2*1j]
#         x = inner_radius * np.cos(u) * np.sin(v)
#         y = inner_radius * np.sin(u) * np.sin(v)
#         z = inner_radius * np.cos(v)
#         return go.Surface(x=x, y=y, z=z, colorscale=[[0, color], [1, color]], 
#                          opacity=opacity, showscale=False)
#     else:  # Shell with random points between inner and outer radius
#         # Generate random points in a shell
#         n_points = 5000
#         phi = np.random.uniform(0, 2*np.pi, n_points)
#         theta = np.arccos(np.random.uniform(-1, 1, n_points))
#         r = np.random.uniform(inner_radius, outer_radius, n_points)
        
#         x = r * np.sin(theta) * np.cos(phi)
#         y = r * np.sin(theta) * np.sin(phi)
#         z = r * np.cos(theta)
        
#         return go.Scatter3d(x=x, y=y, z=z, mode='markers',
#                            marker=dict(size=1.5, color=color, opacity=opacity),
#                            hoverinfo='none', showlegend=True)

# # Helper: create a sphere for a planet
# def create_planet_sphere(x, y, z, radius, color, name, resolution=20):
#     u, v = np.mgrid[0:2*np.pi:resolution*1j, 0:np.pi:resolution//2*1j]
    
#     # Create sphere surface coordinates
#     sphere_x = x + radius * np.cos(u) * np.sin(v)
#     sphere_y = y + radius * np.sin(u) * np.sin(v)
#     sphere_z = z + radius * np.cos(v)
    
#     return go.Surface(
#         x=sphere_x, y=sphere_y, z=sphere_z,
#         colorscale=[[0, color], [1, color]],
#         opacity=1.0, 
#         showscale=False,
#         name=name,
#         hoverinfo='name',
#         lighting=dict(
#             ambient=0.8,  # Increase ambient light to see colors better
#             diffuse=0.9,
#             specular=0.3,
#             roughness=0.5,
#             fresnel=0.2
#         )
#     )

# # Helper: create a disk (for asteroid belt)
# def create_disk(inner_radius, outer_radius, color='gray', opacity=0.2, points=1000):
#     # Create scattered points to represent a disk
#     r = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2, points))
#     theta = np.random.uniform(0, 2*np.pi, points)
    
#     x = r * np.cos(theta)
#     y = r * np.sin(theta)
#     # Add some vertical dispersion for thickness
#     z = np.random.normal(0, (outer_radius-inner_radius)*0.05, points)
    
#     return go.Scatter3d(x=x, y=y, z=z, mode='markers',
#                        marker=dict(size=1.5, color=color, opacity=opacity),
#                        name='Asteroid Belt', hoverinfo='name')

# # Helper: build Plotly traces for a planet list
# def make_traces(planet_list, include_regions=True, use_planet_spheres=True):
#     traces = []
    
#     # Sun sphere
#     u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
#     x_s = 0.05 * np.cos(u) * np.sin(v)
#     y_s = 0.05 * np.sin(u) * np.sin(v)
#     z_s = 0.05 * np.cos(v)
#     traces.append(go.Surface(
#         x=x_s, y=y_s, z=z_s,
#         colorscale=[[0,'yellow'],[1,'yellow']],
#         opacity=0.9, 
#         showscale=False,
#         name="Sun",
#         lighting=dict(
#             ambient=0.9,
#             diffuse=0.8,
#             specular=0.5
#         )
#     ))
    
#     # Add solar system regions if requested
#     if include_regions:
#         # Asteroid Belt (disk representation)
#         ab_params = solar_regions['Asteroid Belt']
#         traces.append(create_disk(ab_params['inner_radius'], ab_params['outer_radius'], 
#                                ab_params['color'], ab_params['opacity']))
        
#         # Kuiper Belt (shell with scattered points)
#         kb_params = solar_regions['Kuiper Belt']
#         kb_trace = create_shell(kb_params['inner_radius'], kb_params['outer_radius'], 
#                               kb_params['color'], kb_params['opacity'])
#         kb_trace.name = 'Kuiper Belt'
#         traces.append(kb_trace)
        
#         # Heliopause (spherical boundary)
#         hp_params = solar_regions['Heliopause']
#         hp_trace = create_shell(hp_params['radius'], None, hp_params['color'], hp_params['opacity'])
#         hp_trace.name = 'Heliopause'
#         traces.append(hp_trace)
    
#     # Planets
#     for i, name in enumerate(planet_list):
#         a, e, inc, Ω, ω, rf = orbital_params[name]
#         inc, Ω, ω = np.radians([inc, Ω, ω])
#         θ = np.linspace(0, 2*np.pi, 500)
#         r = a * (1-e**2) / (1+e*np.cos(θ))
#         x = r*np.cos(θ); y = r*np.sin(θ); z = np.zeros_like(θ)
        
#         # Rotate & incline orbit
#         x1 = x*np.cos(ω)-y*np.sin(ω)
#         y1 = x*np.sin(ω)+y*np.cos(ω)
#         y2 = y1*np.cos(inc); z2 = y1*np.sin(inc)
#         x3 = x1*np.cos(Ω)-y2*np.sin(Ω)
#         y3 = x1*np.sin(Ω)+y2*np.cos(Ω)
        
#         # Orbit line
#         traces.append(go.Scatter3d(
#             x=x3, y=y3, z=z2, 
#             mode='lines',
#             line=dict(color=planet_colors[name], width=2), 
#             showlegend=False
#         ))
        
#         # Calculate current position
#         t0 = np.radians((i*30)%360)
#         r0 = a*(1-e**2)/(1+e*np.cos(t0))
#         x0, y0 = r0*np.cos(t0), r0*np.sin(t0)
#         x0p = x0*np.cos(ω)-y0*np.sin(ω)
#         y0p = x0*np.sin(ω)+y0*np.cos(ω)
#         y0i = y0p*np.cos(inc); z0i = y0p*np.sin(inc)
#         x0r = x0p*np.cos(Ω)-y0i*np.sin(Ω)
#         y0r = x0p*np.sin(Ω)+y0i*np.cos(Ω)
        
#         # Add planet
#         if use_planet_spheres:
#             # Calculate planet size - scale by distance for better visibility
#             # Scale is exaggerated for visualization purposes
#             planet_radius = planet_relative_sizes[name] * size_scale_factor
            
#             # Create sphere for planet
#             traces.append(create_planet_sphere(
#                 x0r, y0r, z0i, 
#                 planet_radius, 
#                 planet_colors[name],
#                 name
#             ))
#         else:
#             # Fall back to simple marker if spheres not requested
#             traces.append(go.Scatter3d(
#                 x=[x0r], y=[y0r], z=[z0i],
#                 mode='markers', 
#                 marker=dict(size=8*rf, color=planet_colors[name]), 
#                 name=name
#             ))
            
#     return traces

# # Helper: build a complete figure
# def build_fig(subset, rng, title, include_regions=True, include_oort=False, use_planet_spheres=True):
#     traces = make_traces(subset, include_regions, use_planet_spheres)
    
#     # Add Oort Cloud separately if requested (only for very large scale views)
#     if include_oort:
#         oc_params = solar_regions['Oort Cloud']
#         oc_trace = create_shell(oc_params['inner_radius'], oc_params['outer_radius'], 
#                               oc_params['color'], oc_params['opacity'])
#         oc_trace.name = 'Oort Cloud'
#         traces.append(oc_trace)
    
#     fig = go.Figure(data=traces)
#     fig.update_layout(
#         scene=dict(
#             xaxis=dict(range=[-rng,rng], title='X (AU)'),
#             yaxis=dict(range=[-rng,rng], title='Y (AU)'),
#             zaxis=dict(range=[-rng,rng], title='Z (AU)'),
#             aspectmode='cube',  # Equal scaling but better use of space
#             camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # Adjust camera position
#         ),
#         margin=dict(l=10, r=10, t=50, b=10),  # Reduced margins
#         title=title,
#         height=700,  # Set explicit height
#         width=700,  # Set explicit width
#         autosize=True,  # Enable autosize
#         legend=dict(
#             yanchor="top",
#             y=0.99,
#             xanchor="left",
#             x=0.01
#         )
#     )
#     return fig

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

# Relative planetary radii (Earth = 1)
planet_relative_sizes = {
    'Mercury': 0.383, 'Venus': 0.950, 'Earth': 1.0, 'Mars': 0.532,
    'Jupiter': 11.21, 'Saturn': 9.45, 'Uranus': 4.01, 'Neptune': 3.88,
    'Ceres': 0.074, 'Pluto': 0.186, 'Eris': 0.183, 'Haumea': 0.16,
    'Makemake': 0.18, 'Sedna': 0.10
}

# Scale factor to make planets more visible while maintaining relative proportions
size_scale_factor = 0.15

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

# Helper: create a see-through ellipsoid
def create_ellipsoid(inner_edge=0, length=950, width_factor=0.75, height_factor=0.5, color='rgba(255,100,100,0.3)', opacity=0.2, name="Extended Region", points=40):
    """
    Create a see-through ellipsoid with left edge at a specified distance, extending opposite from the sun.
    
    Parameters:
    - inner_edge: Distance from sun to the inner edge (AU)
    - length: Length of the ellipsoid extending away from the sun (AU)
    - width_factor, height_factor: Factors to determine the width and height relative to length
    - color: Color of the ellipsoid
    - opacity: Opacity of the ellipsoid
    - name: Name for the hover label
    - points: Resolution of the ellipsoid mesh
    
    Returns:
    - A Plotly Surface trace object
    """
    # Calculate ellipsoid parameters (center is shifted from origin)
    center_x = inner_edge + length/2  # Center is between inner edge and total extent
    a = length/2  # Semi-major axis (half length)
    b = length * width_factor/2  # Semi-minor axis (half width)
    c = length * height_factor/2  # Semi-minor axis (half height)
    
    # Create parametric surface
    u, v = np.mgrid[0:2*np.pi:points*1j, 0:np.pi:points//2*1j]
    
    # Ellipsoid surface coordinates 
    x = center_x + a * np.cos(u) * np.sin(v)  # Note the + center_x to shift away from origin
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)
    
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, color]],
        opacity=opacity, 
        showscale=False,
        name=name,
        hoverinfo='name',
        lighting=dict(
            ambient=0.7,
            diffuse=0.8,
            specular=0.2,
            roughness=0.7,
            fresnel=0.2
        )
    )

# Helper: create a sphere for a planet
def create_planet_sphere(x, y, z, radius, color, name, resolution=20):
    u, v = np.mgrid[0:2*np.pi:resolution*1j, 0:np.pi:resolution//2*1j]
    
    # Create sphere surface coordinates
    sphere_x = x + radius * np.cos(u) * np.sin(v)
    sphere_y = y + radius * np.sin(u) * np.sin(v)
    sphere_z = z + radius * np.cos(v)
    
    return go.Surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        colorscale=[[0, color], [1, color]],
        opacity=1.0, 
        showscale=False,
        name=name,
        hoverinfo='name',
        lighting=dict(
            ambient=0.8,  # Increase ambient light to see colors better
            diffuse=0.9,
            specular=0.3,
            roughness=0.5,
            fresnel=0.2
        )
    )

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
def make_traces(planet_list, include_regions=True, use_planet_spheres=True, include_ellipsoid=False, ellipsoid_params=None):
    traces = []
    
    # Sun sphere
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x_s = 0.05 * np.cos(u) * np.sin(v)
    y_s = 0.05 * np.sin(u) * np.sin(v)
    z_s = 0.05 * np.cos(v)
    traces.append(go.Surface(
        x=x_s, y=y_s, z=z_s,
        colorscale=[[0,'yellow'],[1,'yellow']],
        opacity=0.9, 
        showscale=False,
        name="Sun",
        lighting=dict(
            ambient=0.9,
            diffuse=0.8,
            specular=0.5
        )
    ))
    
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
    
    # Add custom ellipsoid if requested
    if include_ellipsoid:
        # Use default parameters if none provided
        if ellipsoid_params is None:
            ellipsoid_params = {
                'inner_edge': 120,
                'length': 300,
                'width_factor': 0.75,
                'height_factor': 0.5,
                'color': 'rgba(255,100,100,0.3)',
                'opacity': 0.2,
                'name': 'Extended Region'
            }
        
        traces.append(create_ellipsoid(**ellipsoid_params))
    
    # Planets
    for i, name in enumerate(planet_list):
        a, e, inc, Ω, ω, rf = orbital_params[name]
        inc, Ω, ω = np.radians([inc, Ω, ω])
        θ = np.linspace(0, 2*np.pi, 500)
        r = a * (1-e**2) / (1+e*np.cos(θ))
        x = r*np.cos(θ); y = r*np.sin(θ); z = np.zeros_like(θ)
        
        # Rotate & incline orbit
        x1 = x*np.cos(ω)-y*np.sin(ω)
        y1 = x*np.sin(ω)+y*np.cos(ω)
        y2 = y1*np.cos(inc); z2 = y1*np.sin(inc)
        x3 = x1*np.cos(Ω)-y2*np.sin(Ω)
        y3 = x1*np.sin(Ω)+y2*np.cos(Ω)
        
        # Orbit line
        traces.append(go.Scatter3d(
            x=x3, y=y3, z=z2, 
            mode='lines',
            line=dict(color=planet_colors[name], width=2), 
            showlegend=False
        ))
        
        # Calculate current position
        t0 = np.radians((i*30)%360)
        r0 = a*(1-e**2)/(1+e*np.cos(t0))
        x0, y0 = r0*np.cos(t0), r0*np.sin(t0)
        x0p = x0*np.cos(ω)-y0*np.sin(ω)
        y0p = x0*np.sin(ω)+y0*np.cos(ω)
        y0i = y0p*np.cos(inc); z0i = y0p*np.sin(inc)
        x0r = x0p*np.cos(Ω)-y0i*np.sin(Ω)
        y0r = x0p*np.sin(Ω)+y0i*np.cos(Ω)
        
        # Add planet
        if use_planet_spheres:
            # Calculate planet size - scale by distance for better visibility
            # Scale is exaggerated for visualization purposes
            planet_radius = planet_relative_sizes[name] * size_scale_factor
            
            # Create sphere for planet
            traces.append(create_planet_sphere(
                x0r, y0r, z0i, 
                planet_radius, 
                planet_colors[name],
                name
            ))
        else:
            # Fall back to simple marker if spheres not requested
            traces.append(go.Scatter3d(
                x=[x0r], y=[y0r], z=[z0i],
                mode='markers', 
                marker=dict(size=8*rf, color=planet_colors[name]), 
                name=name
            ))
            
    return traces

# Helper: build a complete figure
def build_fig(subset, rng, title, include_regions=True, include_oort=False, use_planet_spheres=True, include_ellipsoid=False, ellipsoid_params=None):
    traces = make_traces(subset, include_regions, use_planet_spheres, include_ellipsoid, ellipsoid_params)
    
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
            aspectmode='cube',  # Equal scaling but better use of space
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