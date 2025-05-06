# solar_helpers_galactic.py
import numpy as np
import plotly.graph_objects as go

# Shared data (unchanged from original)
planet_colors = {
    'Mercury': '#8c8680', 'Venus': '#e6c89c', 'Earth': '#4f71be', 'Mars': '#d1603d',
    'Jupiter': '#e0ae6f', 'Saturn': '#c5ab6e', 'Uranus': '#9fc4e7', 'Neptune': '#4f71be',
    'Ceres': '#8c8680', 'Pluto': '#ab9c8a', 'Eris': '#d9d9d9', 'Haumea': '#d9d9d9',
    'Makemake': '#c49e6c', 'Sedna': '#bb5540', 'Gonggong': '#bb6a50', 'Quaoar': '#b79e85',
    'Orcus': '#bfb8b0'
}

# Relative planetary radii (Earth = 1)
planet_relative_sizes = {
    'Mercury': 0.383, 'Venus': 0.950, 'Earth': 1.0, 'Mars': 0.532,
    'Jupiter': 11.21, 'Saturn': 9.45, 'Uranus': 4.01, 'Neptune': 3.88,
    'Ceres': 0.074, 'Pluto': 0.186, 'Eris': 0.183, 'Haumea': 0.16,
    'Makemake': 0.18, 'Sedna': 0.10, 'Gonggong': 0.12, 'Quaoar': 0.09, 'Orcus': 0.09
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
    'Sedna':   [506,   0.8459, 11.93,    144.31,     311.46,    0.006],
    'Gonggong':[67.5,  0.5,    30.7,     101.2,      208.1,     0.007],
    'Quaoar':  [43.7,  0.04,   8.0,      188.8,      25.2,      0.006],
    'Orcus':   [39.42, 0.227,  20.6,     268.8,      73.12,     0.006]
}

# Solar system region parameters (AU)
solar_regions = {
    'Asteroid Belt': {'inner_radius': 2.2, 'outer_radius': 3.2, 'color': 'gray', 'opacity': 0.2},
    'Kuiper Belt': {'inner_radius': 30, 'outer_radius': 50, 'color': 'lightblue', 'opacity': 0.15},
    'Heliopause': {'radius': 120, 'color': 'rgba(100,100,255,0.5)', 'opacity': 0.1},
    'Oort Cloud': {'inner_radius': 2000, 'outer_radius': 100000, 'color': 'rgba(200,200,255,0.3)', 'opacity': 0.05}
}

# NEW: Galactic Coordinate System class
class GalacticCoordinates:
    """
    Handles transformations between Solar System coordinates (ecliptic) and Galactic coordinates.
    """
    
    def __init__(self):
        # J2000 epoch values
        # Galactic North Pole in ICRS (equatorial) coordinates
        self.pole_ra = np.radians(192.859508)  # Right ascension of Galactic North Pole
        self.pole_dec = np.radians(27.128336)  # Declination of Galactic North Pole
        self.l0 = np.radians(122.932)  # Galactic longitude of the celestial North Pole
        
        # Pre-compute transformation matrices
        self._compute_transformation_matrices()
    
    def _compute_transformation_matrices(self):
        """Compute the rotation matrices for coordinate transformations."""
        # Rotation matrix from equatorial to Galactic
        sin_pole_ra = np.sin(self.pole_ra)
        cos_pole_ra = np.cos(self.pole_ra)
        sin_pole_dec = np.sin(self.pole_dec)
        cos_pole_dec = np.cos(self.pole_dec)
        sin_l0 = np.sin(self.l0)
        cos_l0 = np.cos(self.l0)
        
        # Create the transformation matrix from equatorial to Galactic
        self.eq_to_gal = np.array([
            [-sin_pole_ra * sin_l0 - cos_pole_ra * sin_pole_dec * cos_l0, 
             cos_pole_ra * sin_l0 - sin_pole_ra * sin_pole_dec * cos_l0, 
             -cos_pole_dec * cos_l0],
            [sin_pole_ra * cos_l0 - cos_pole_ra * sin_pole_dec * sin_l0, 
             -cos_pole_ra * cos_l0 - sin_pole_ra * sin_pole_dec * sin_l0, 
             -cos_pole_dec * sin_l0],
            [cos_pole_ra * cos_pole_dec, sin_pole_ra * cos_pole_dec, sin_pole_dec]
        ])
        
        # Galactic to equatorial is the transpose (inverse) of the above matrix
        self.gal_to_eq = self.eq_to_gal.T
        
        # Pre-compute the obliquity of the ecliptic (J2000)
        self.epsilon = np.radians(23.4392911)
        
    def ecliptic_to_equatorial(self, x, y, z):
        """
        Convert from ecliptic to equatorial coordinates.
        
        Parameters:
        -----------
        x, y, z : float or array
            Coordinates in the ecliptic system (AU)
        
        Returns:
        --------
        x_eq, y_eq, z_eq : float or array
            Coordinates in the equatorial system (AU)
        """
        # Transformation
        x_eq = x
        y_eq = y * np.cos(self.epsilon) - z * np.sin(self.epsilon)
        z_eq = y * np.sin(self.epsilon) + z * np.cos(self.epsilon)
        
        return x_eq, y_eq, z_eq
    
    def equatorial_to_galactic(self, x_eq, y_eq, z_eq):
        """
        Convert from equatorial to Galactic coordinates.
        
        Parameters:
        -----------
        x_eq, y_eq, z_eq : float or array
            Coordinates in the equatorial system (AU)
        
        Returns:
        --------
        x_gal, y_gal, z_gal : float or array
            Coordinates in the Galactic system (AU)
        """
        if isinstance(x_eq, np.ndarray):
            # Handle arrays using matrix multiplication
            coords_eq = np.vstack((x_eq, y_eq, z_eq))
            coords_gal = self.eq_to_gal @ coords_eq
            return coords_gal[0, :], coords_gal[1, :], coords_gal[2, :]
        else:
            # Handle single points
            coords_eq = np.array([x_eq, y_eq, z_eq])
            coords_gal = self.eq_to_gal @ coords_eq
            return coords_gal[0], coords_gal[1], coords_gal[2]
    
    def ecliptic_to_galactic(self, x, y, z):
        """
        Convert directly from ecliptic to Galactic coordinates.
        
        Parameters:
        -----------
        x, y, z : float or array
            Coordinates in the ecliptic system (AU)
        
        Returns:
        --------
        x_gal, y_gal, z_gal : float or array
            Coordinates in the Galactic system (AU)
        """
        # First convert to equatorial
        x_eq, y_eq, z_eq = self.ecliptic_to_equatorial(x, y, z)
        
        # Then convert to Galactic
        return self.equatorial_to_galactic(x_eq, y_eq, z_eq)
    
    def galactic_to_equatorial(self, x_gal, y_gal, z_gal):
        """
        Convert from Galactic to equatorial coordinates.
        
        Parameters:
        -----------
        x_gal, y_gal, z_gal : float or array
            Coordinates in the Galactic system (AU)
        
        Returns:
        --------
        x_eq, y_eq, z_eq : float or array
            Coordinates in the equatorial system (AU)
        """
        if isinstance(x_gal, np.ndarray):
            # Handle arrays
            coords_gal = np.vstack((x_gal, y_gal, z_gal))
            coords_eq = self.gal_to_eq @ coords_gal
            return coords_eq[0, :], coords_eq[1, :], coords_eq[2, :]
        else:
            # Handle single points
            coords_gal = np.array([x_gal, y_gal, z_gal])
            coords_eq = self.gal_to_eq @ coords_gal
            return coords_eq[0], coords_eq[1], coords_eq[2]
    
    def equatorial_to_ecliptic(self, x_eq, y_eq, z_eq):
        """
        Convert from equatorial to ecliptic coordinates.
        
        Parameters:
        -----------
        x_eq, y_eq, z_eq : float or array
            Coordinates in the equatorial system (AU)
        
        Returns:
        --------
        x, y, z : float or array
            Coordinates in the ecliptic system (AU)
        """
        # Transformation
        x = x_eq
        y = y_eq * np.cos(self.epsilon) + z_eq * np.sin(self.epsilon)
        z = -y_eq * np.sin(self.epsilon) + z_eq * np.cos(self.epsilon)
        
        return x, y, z
    
    def galactic_to_ecliptic(self, x_gal, y_gal, z_gal):
        """
        Convert directly from Galactic to ecliptic coordinates.
        
        Parameters:
        -----------
        x_gal, y_gal, z_gal : float or array
            Coordinates in the Galactic system (AU)
        
        Returns:
        --------
        x, y, z : float or array
            Coordinates in the ecliptic system (AU)
        """
        # First convert to equatorial
        x_eq, y_eq, z_eq = self.galactic_to_equatorial(x_gal, y_gal, z_gal)
        
        # Then convert to ecliptic
        return self.equatorial_to_ecliptic(x_eq, y_eq, z_eq)


# Instantiate the coordinate converter
galactic_converter = GalacticCoordinates()

# Helper: create a spherical shell (updated for galactic coordinate option)
def create_shell(inner_radius, outer_radius=None, color='gray', opacity=0.2, points=40, use_galactic=False):
    if outer_radius is None:  # Simple sphere
        u, v = np.mgrid[0:2*np.pi:points*1j, 0:np.pi:points//2*1j]
        x = inner_radius * np.cos(u) * np.sin(v)
        y = inner_radius * np.sin(u) * np.sin(v)
        z = inner_radius * np.cos(v)
        
        # Convert to galactic coordinates if requested
        if use_galactic:
            x, y, z = galactic_converter.ecliptic_to_galactic(x, y, z)
            
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
        
        # Convert to galactic coordinates if requested
        if use_galactic:
            x, y, z = galactic_converter.ecliptic_to_galactic(x, y, z)
            
        return go.Scatter3d(x=x, y=y, z=z, mode='markers',
                           marker=dict(size=1.5, color=color, opacity=opacity),
                           hoverinfo='none', showlegend=True)

# Helper: create a see-through ellipsoid (updated for galactic coordinate option)
def create_ellipsoid(inner_edge=0, length=950, width_factor=0.75, height_factor=0.5, 
                    color='rgba(255,100,100,0.3)', opacity=0.2, name="Extended Region", 
                    points=40, use_galactic=False):
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
    - use_galactic: Whether to convert to galactic coordinates
    
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
    
    # Convert to galactic coordinates if requested
    if use_galactic:
        x, y, z = galactic_converter.ecliptic_to_galactic(x, y, z)
    
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

# Helper: create a sphere for a planet (updated for galactic coordinate option)
def create_planet_sphere(x, y, z, radius, color, name, resolution=20, use_galactic=False):
    u, v = np.mgrid[0:2*np.pi:resolution*1j, 0:np.pi:resolution//2*1j]
    
    # Convert center to galactic coordinates if requested
    if use_galactic:
        x, y, z = galactic_converter.ecliptic_to_galactic(x, y, z)
    
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

# Helper: create a disk (for asteroid belt) (updated for galactic coordinate option)
def create_disk(inner_radius, outer_radius, color='gray', opacity=0.2, points=1000, use_galactic=False):
    # Create scattered points to represent a disk
    r = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2, points))
    theta = np.random.uniform(0, 2*np.pi, points)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # Add some vertical dispersion for thickness
    z = np.random.normal(0, (outer_radius-inner_radius)*0.05, points)
    
    # Convert to galactic coordinates if requested
    if use_galactic:
        x, y, z = galactic_converter.ecliptic_to_galactic(x, y, z)
    
    return go.Scatter3d(x=x, y=y, z=z, mode='markers',
                       marker=dict(size=1.5, color=color, opacity=opacity),
                       name='Asteroid Belt', hoverinfo='name')

# Helper: build Plotly traces for a planet list (updated for galactic coordinate option)
def make_traces(planet_list, include_regions=True, use_planet_spheres=True, 
               include_ellipsoid=False, ellipsoid_params=None, use_galactic=False):
    traces = []
    
    # Sun sphere
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x_s = 0.05 * np.cos(u) * np.sin(v)
    y_s = 0.05 * np.sin(u) * np.sin(v)
    z_s = 0.05 * np.cos(v)
    
    # Convert to galactic coordinates if requested
    if use_galactic:
        x_s, y_s, z_s = galactic_converter.ecliptic_to_galactic(x_s, y_s, z_s)
    
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
                                ab_params['color'], ab_params['opacity'], use_galactic=use_galactic))
        
        # Kuiper Belt (shell with scattered points)
        kb_params = solar_regions['Kuiper Belt']
        kb_trace = create_shell(kb_params['inner_radius'], kb_params['outer_radius'], 
                              kb_params['color'], kb_params['opacity'], use_galactic=use_galactic)
        kb_trace.name = 'Kuiper Belt'
        traces.append(kb_trace)
        
        # Heliopause (spherical boundary)
        hp_params = solar_regions['Heliopause']
        hp_trace = create_shell(hp_params['radius'], None, hp_params['color'], 
                             hp_params['opacity'], use_galactic=use_galactic)
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
        
        traces.append(create_ellipsoid(**ellipsoid_params, use_galactic=use_galactic))
    
    # Galactic coordinate reference axes (when in galactic mode)
    if use_galactic:
        # Axis length in AU
        axis_length = 5.0
        
        # Galactic center direction (X)
        traces.append(go.Scatter3d(
            x=[0, axis_length], y=[0, 0], z=[0, 0],
            mode='lines',
            line=dict(color='red', width=5),
            name='Galactic Center Direction',
            showlegend=True
        ))
        
        # Galactic rotation direction (Y)
        traces.append(go.Scatter3d(
            x=[0, 0], y=[0, axis_length], z=[0, 0],
            mode='lines',
            line=dict(color='green', width=5),
            name='Galactic Rotation Direction',
            showlegend=True
        ))
        
        # North Galactic Pole (Z)
        traces.append(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, axis_length],
            mode='lines',
            line=dict(color='blue', width=5),
            name='North Galactic Pole',
            showlegend=True
        ))
    
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
        
        # Convert orbit to galactic coordinates if requested
        if use_galactic:
            x3, y3, z2 = galactic_converter.ecliptic_to_galactic(x3, y3, z2)
        
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
            
            # Create sphere for planet (with galactic conversion if needed)
            traces.append(create_planet_sphere(
                x0r, y0r, z0i, 
                planet_radius, 
                planet_colors[name],
                name,
                use_galactic=use_galactic
            ))
        else:
            # Fall back to simple marker if spheres not requested
            # Convert position to galactic coordinates if requested
            if use_galactic:
                x0r, y0r, z0i = galactic_converter.ecliptic_to_galactic(x0r, y0r, z0i)
                
            traces.append(go.Scatter3d(
                x=[x0r], y=[y0r], z=[z0i],
                mode='markers', 
                marker=dict(size=8*rf, color=planet_colors[name]), 
                name=name
            ))
            
    return traces

# Helper: build a complete figure (updated for galactic coordinate option)
def build_fig(subset, rng, title, include_regions=True, include_oort=False, 
             use_planet_spheres=True, include_ellipsoid=False, ellipsoid_params=None, 
             use_galactic=False):
    
    traces = make_traces(
        subset, include_regions, use_planet_spheres, 
        include_ellipsoid, ellipsoid_params, use_galactic
    )
    
    # Add Oort Cloud separately if requested (only for very large scale views)
    if include_oort:
        oc_params = solar_regions['Oort Cloud']
        oc_trace = create_shell(
            oc_params['inner_radius'], oc_params['outer_radius'], 
            oc_params['color'], oc_params['opacity'], use_galactic=use_galactic
        )
        oc_trace.name = 'Oort Cloud'
        traces.append(oc_trace)
    
    # Set up axis labels based on coordinate system
    if use_galactic:
        x_label = 'X (Galactic Center Direction, AU)'
        y_label = 'Y (Galactic Rotation Direction, AU)'
        z_label = 'Z (North Galactic Pole, AU)'
        # Update title to indicate galactic coordinates
        title = f"{title} (Galactic Coordinates)"
    else:
        x_label = 'X (AU)'
        y_label = 'Y (AU)'
        z_label = 'Z (AU)'
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-rng,rng], title=x_label),
            yaxis=dict(range=[-rng,rng], title=y_label),
            zaxis=dict(range=[-rng,rng], title=z_label),
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

# Completion of example_galactic_comparison function and additional utility functions

def example_galactic_comparison():
    """Create a side-by-side comparison of ecliptic and galactic coordinates"""
    from plotly.subplots import make_subplots
    
    # Define planets to include
    planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    
    # Create individual figures
    fig_ecliptic = build_fig(planets, 35, "Solar System", use_galactic=False)
    fig_galactic = build_fig(planets, 35, "Solar System", use_galactic=True)
    
    # Create a subplot figure
    subplot_fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Ecliptic Coordinates", "Galactic Coordinates")
    )
    
    # Add traces from individual figures to subplot
    for trace in fig_ecliptic.data:
        subplot_fig.add_trace(trace, row=1, col=1)
        
    for trace in fig_galactic.data:
        subplot_fig.add_trace(trace, row=1, col=2)
    
    # Update layout for the combined figure
    subplot_fig.update_layout(
        width=1400,    # Double width for side-by-side
        height=700,
        title_text="Solar System: Ecliptic vs Galactic Coordinate Systems",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Ensure proper scene aspects
    subplot_fig.update_scenes(
        aspectmode='cube',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    )
    
    return subplot_fig

# Additional useful functions

def galactic_position_table(planet_list):
    """
    Generate a table of planetary positions in both ecliptic and galactic coordinates
    
    Parameters:
    -----------
    planet_list : list
        List of planet names to include in the table
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the position data
    """
    import pandas as pd
    
    # Initialize lists to store data
    names = []
    x_ecl, y_ecl, z_ecl = [], [], []
    x_gal, y_gal, z_gal = [], [], []
    
    # Calculate positions for each planet
    for i, name in enumerate(planet_list):
        # Extract orbital parameters
        a, e, inc, Ω, ω, _ = orbital_params[name]
        inc, Ω, ω = np.radians([inc, Ω, ω])
        
        # Calculate position in orbit (staggered positions)
        t0 = np.radians((i*30)%360)
        r0 = a*(1-e**2)/(1+e*np.cos(t0))
        x0, y0 = r0*np.cos(t0), r0*np.sin(t0)
        
        # Apply orbital transformations
        x0p = x0*np.cos(ω)-y0*np.sin(ω)
        y0p = x0*np.sin(ω)+y0*np.cos(ω)
        y0i = y0p*np.cos(inc)
        z0i = y0p*np.sin(inc)
        x0r = x0p*np.cos(Ω)-y0i*np.sin(Ω)
        y0r = x0p*np.sin(Ω)+y0i*np.cos(Ω)
        
        # Convert to galactic coordinates
        x0g, y0g, z0g = galactic_converter.ecliptic_to_galactic(x0r, y0r, z0i)
        
        # Store data
        names.append(name)
        x_ecl.append(x0r)
        y_ecl.append(y0r)
        z_ecl.append(z0i)
        x_gal.append(x0g)
        y_gal.append(y0g)
        z_gal.append(z0g)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Planet': names,
        'X_Ecliptic (AU)': x_ecl,
        'Y_Ecliptic (AU)': y_ecl,
        'Z_Ecliptic (AU)': z_ecl,
        'X_Galactic (AU)': x_gal,
        'Y_Galactic (AU)': y_gal,
        'Z_Galactic (AU)': z_gal
    })
    
    # Format to 3 decimal places
    for col in df.columns:
        if col != 'Planet':
            df[col] = df[col].map(lambda x: f"{x:.3f}")
    
    return df

def create_galactic_plane_marker(size=100, opacity=0.1, use_galactic=True):
    """
    Create a representation of the galactic plane
    
    Parameters:
    -----------
    size : float
        Size of the galactic plane marker in AU
    opacity : float
        Opacity of the galactic plane
    use_galactic : bool
        Whether to use galactic coordinates (should typically be True)
        
    Returns:
    --------
    plotly.graph_objects.Surface
        Surface object representing the galactic plane
    """
    # Create a disk representing the galactic plane
    u, v = np.mgrid[0:1:20j, 0:2*np.pi:30j]
    r = size * u
    
    # In galactic coordinates, the plane is the XY plane
    x = r * np.cos(v)
    y = r * np.sin(v)
    z = np.zeros_like(x)
    
    # If not in galactic mode, transform back to ecliptic
    if not use_galactic:
        x, y, z = galactic_converter.galactic_to_ecliptic(x, y, z)
    
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, 'rgba(100,50,200,0.5)'], [1, 'rgba(200,150,255,0.5)']],
        opacity=opacity,
        showscale=False,
        name="Galactic Plane",
        hoverinfo='name'
    )

def solar_system_with_galactic_plane(planet_list, range_value=50, include_regions=True):
    """
    Create a solar system visualization that includes the galactic plane
    
    Parameters:
    -----------
    planet_list : list
        List of planets to include
    range_value : float
        Range for the axis limits (AU)
    include_regions : bool
        Whether to include solar system regions
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figure containing the visualization
    """
    # Get the basic solar system figure
    fig = build_fig(planet_list, range_value, "Solar System with Galactic Plane",
                   include_regions=include_regions, use_galactic=False)
    
    # Add the galactic plane
    galactic_plane = create_galactic_plane_marker(size=range_value*0.95, opacity=0.15, use_galactic=False)
    fig.add_trace(galactic_plane)
    
    # Add a text annotation for the galactic center direction
    # Convert galactic center direction to ecliptic coordinates
    gc_x, gc_y, gc_z = galactic_converter.galactic_to_ecliptic(range_value, 0, 0)
    gc_direction = go.Scatter3d(
        x=[0, gc_x], y=[0, gc_y], z=[0, gc_z],
        mode='lines',
        line=dict(color='purple', width=3, dash='dash'),
        name='Galactic Center Direction',
        showlegend=True
    )
    fig.add_trace(gc_direction)
    
    # Update layout to indicate special features
    fig.update_layout(
        title="Solar System with Galactic Plane (Ecliptic Coordinates)",
        scene=dict(
            annotations=[
                dict(
                    x=gc_x,
                    y=gc_y,
                    z=gc_z,
                    text="Galactic Center Direction",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='purple'
                )
            ]
        )
    )
    
    return fig

def galactic_orientation_visualization(size=10):
    """
    Create a simple visualization to show the orientation of the galactic coordinate system
    relative to the ecliptic system
    
    Parameters:
    -----------
    size : float
        Size of the visualization in AU
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figure showing the relative orientation
    """
    traces = []
    
    # Create ecliptic axes
    for axis, color, name in zip(
        [(size,0,0), (0,size,0), (0,0,size)],
        ['red', 'green', 'blue'],
        ['Ecliptic X', 'Ecliptic Y', 'Ecliptic Z']
    ):
        traces.append(go.Scatter3d(
            x=[0, axis[0]], y=[0, axis[1]], z=[0, axis[2]],
            mode='lines',
            line=dict(color=color, width=5),
            name=name
        ))
    
    # Create galactic axes transformed to ecliptic
    gal_x_ecl = galactic_converter.galactic_to_ecliptic(size, 0, 0)
    gal_y_ecl = galactic_converter.galactic_to_ecliptic(0, size, 0)
    gal_z_ecl = galactic_converter.galactic_to_ecliptic(0, 0, size)
    
    for axis, color, name in zip(
        [gal_x_ecl, gal_y_ecl, gal_z_ecl],
        ['darkred', 'darkgreen', 'darkblue'],
        ['Galactic X (Center)', 'Galactic Y (Rotation)', 'Galactic Z (Pole)']
    ):
        traces.append(go.Scatter3d(
            x=[0, axis[0]], y=[0, axis[1]], z=[0, axis[2]],
            mode='lines',
            line=dict(color=color, width=5, dash='dash'),
            name=name
        ))
    
    # Create figure
    fig = go.Figure(data=traces)
    
    # Add annotations for clarity
    annotations = []
    for axis, text in zip(
        [(size,0,0), (0,size,0), (0,0,size), gal_x_ecl, gal_y_ecl, gal_z_ecl],
        ['Ecliptic X', 'Ecliptic Y', 'Ecliptic Z', 'Galactic Center', 'Galactic Rotation', 'Galactic Pole']
    ):
        annotations.append(
            dict(
                x=axis[0], y=axis[1], z=axis[2],
                text=text,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1
            )
        )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-size, size], title='X (AU)'),
            yaxis=dict(range=[-size, size], title='Y (AU)'),
            zaxis=dict(range=[-size, size], title='Z (AU)'),
            aspectmode='cube',
            annotations=annotations
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        title="Ecliptic vs Galactic Coordinate System Orientation",
        height=800,
        width=800
    )
    
    return fig

# Example usage function for interactive demos
def interactive_galactic_view(initial_planets=None):
    """
    Create an interactive demo with ipywidgets for exploring the galactic view
    
    Parameters:
    -----------
    initial_planets : list, optional
        Initial list of planets to show
        
    Returns:
    --------
    ipywidgets.VBox
        Interactive widget for exploring the visualization
    """
    import ipywidgets as widgets
    from IPython.display import display
    
    if initial_planets is None:
        initial_planets = ['Mercury', 'Venus', 'Earth', 'Mars']
    
    # All available objects
    all_planets = list(orbital_params.keys())
    
    # Create widgets
    planet_checkboxes = {planet: widgets.Checkbox(
        value=planet in initial_planets,
        description=planet,
        disabled=False
    ) for planet in all_planets}
    
    coord_toggle = widgets.RadioButtons(
        options=['Ecliptic', 'Galactic', 'Side by Side'],
        value='Ecliptic',
        description='Coordinates:',
        disabled=False
    )
    
    range_slider = widgets.FloatSlider(
        value=10.0,
        min=1.0,
        max=100.0,
        step=1.0,
        description='Range (AU):',
        disabled=False,
        continuous_update=False
    )
    
    regions_toggle = widgets.Checkbox(
        value=True,
        description='Show Regions',
        disabled=False
    )
    
    show_galactic_plane = widgets.Checkbox(
        value=False,
        description='Show Galactic Plane',
        disabled=False
    )
    
    output = widgets.Output()
    
    # Create layout
    planets_group = widgets.VBox([widgets.Label('Select Planets:')] + 
                                 [planet_checkboxes[p] for p in all_planets])
    controls = widgets.VBox([
        coord_toggle,
        range_slider,
        regions_toggle,
        show_galactic_plane
    ])
    
    # Create update function
    def update_plot(*args):
        with output:
            output.clear_output(wait=True)
            
            # Get selected planets
            selected_planets = [planet for planet, checkbox in planet_checkboxes.items() 
                               if checkbox.value]
            
            if not selected_planets:
                print("Please select at least one planet.")
                return
            
            # Create appropriate figure based on selection
            if coord_toggle.value == 'Side by Side':
                fig = example_galactic_comparison()
            else:
                use_galactic = coord_toggle.value == 'Galactic'
                fig = build_fig(
                    selected_planets, 
                    range_slider.value,
                    f"Solar System ({coord_toggle.value} Coordinates)",
                    include_regions=regions_toggle.value,
                    use_galactic=use_galactic
                )
                
                # Add galactic plane if requested
                if show_galactic_plane.value:
                    galactic_plane = create_galactic_plane_marker(
                        size=range_slider.value*0.95, 
                        opacity=0.15, 
                        use_galactic=use_galactic
                    )
                    fig.add_trace(galactic_plane)
            
            display(fig)
    
    # Connect widgets to update function
    for checkbox in planet_checkboxes.values():
        checkbox.observe(update_plot, 'value')
    coord_toggle.observe(update_plot, 'value')
    range_slider.observe(update_plot, 'value')
    regions_toggle.observe(update_plot, 'value')
    show_galactic_plane.observe(update_plot, 'value')
    
    # Initial update
    update_plot()
    
    # Create final layout
    return widgets.VBox([
        widgets.HBox([planets_group, controls]),
        output
    ])

# Example of how to use these functions in a Jupyter notebook:
"""
from solar_helpers_galactic import *

# Compare ecliptic and galactic views side by side
example_galactic_comparison()

# View the solar system with the galactic plane
solar_system_with_galactic_plane(['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn'], 15)

# See the orientation of coordinate systems
galactic_orientation_visualization()

# Get a table of planetary positions in both coordinate systems
galactic_position_table(['Mercury', 'Venus', 'Earth', 'Mars'])

# For interactive exploration (in Jupyter notebook)
interactive_galactic_view()
"""