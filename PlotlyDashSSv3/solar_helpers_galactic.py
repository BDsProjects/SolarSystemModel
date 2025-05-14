# solar_helpers_galactic.py
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math

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


# Add time calculation functions at the beginning
def julian_date(date):
    """
    Calculate Julian Date from a datetime object.
    
    Parameters:
    -----------
    date : datetime
        The date to convert
        
    Returns:
    --------
    float
        Julian date
    """
    a = (14 - date.month) // 12
    y = date.year + 4800 - a
    m = date.month + 12 * a - 3
    
    jd = date.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd += (date.hour + date.minute/60.0 + date.second/3600.0) / 24.0
    
    return jd

def julian_centuries_since_j2000(jd):
    """
    Calculate centuries since J2000.0 epoch.
    
    Parameters:
    -----------
    jd : float
        Julian date
        
    Returns:
    --------
    float
        Centuries since J2000.0
    """
    j2000 = 2451545.0  # Julian date for J2000.0 epoch
    return (jd - j2000) / 36525.0

# Enhanced orbital parameters with mean motion and epoch elements
orbital_params_time = {
    # Name: [a(AU), e, i(deg), Ω(deg), ω(deg), M0(deg), n(deg/day)]
    # Where M0 is mean anomaly at epoch J2000.0, n is mean motion
    'Mercury': [0.387, 0.2056, 7.005, 48.331, 29.124, 174.796, 4.09233445],
    'Venus':   [0.723, 0.0068, 3.39458, 76.680, 54.884, 50.115, 1.60213034],
    'Earth':   [1.0,   0.0167, 0.00005, -11.26064, 102.94719, 357.529, 0.98560028],
    'Mars':    [1.524, 0.0934, 1.850, 49.558, 286.502, 19.373, 0.52402068],
    'Jupiter': [5.2,   0.0489, 1.303, 100.464, 273.867, 20.020, 0.08308529],
    'Saturn':  [9.58,  0.0565, 2.485, 113.665, 339.392, 317.020, 0.03344414],
    'Uranus':  [19.22, 0.0457, 0.773, 74.006, 96.998, 142.238, 0.01172834],
    'Neptune': [30.05, 0.0113, 1.77, 131.783, 273.187, 256.228, 0.00598927],
    'Ceres':   [2.77,  0.0758, 10.593, 80.393, 73.597, 291.428, 0.21411192],
    'Pluto':   [39.48, 0.2488, 17.16, 110.299, 113.763, 14.882, 0.00397671],
    'Eris':    [67.8,  0.44068, 44.04, 35.95, 151.639, 204.16, 0.00176901],
    'Haumea':  [43.13, 0.19126, 28.19, 121.9, 239, 293.42, 0.00321123],
    'Makemake':[45.79, 0.159, 29, 79, 296, 85.13, 0.00292961],
    'Sedna':   [506,   0.8459, 11.93, 144.31, 311.46, 358.62, 0.00007627],
    'Gonggong':[67.5,  0.5, 30.7, 101.2, 208.1, 245.8, 0.00177736],
    'Quaoar':  [43.7,  0.04, 8.0, 188.8, 25.2, 330.9, 0.00314073],
    'Orcus':   [39.42, 0.227, 20.6, 268.8, 73.12, 71.93, 0.00398887]
}

def solve_kepler_equation(M, e, tolerance=1e-8):
    """
    Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E.
    
    Parameters:
    -----------
    M : float
        Mean anomaly in radians
    e : float
        Eccentricity
    tolerance : float
        Convergence tolerance
        
    Returns:
    --------
    float
        Eccentric anomaly in radians
    """
    E = M  # Initial guess
    while True:
        E_new = M + e * np.sin(E)
        if abs(E_new - E) < tolerance:
            break
        E = E_new
    return E

def calculate_orbital_position(planet_name, time, use_time=True):
    """
    Calculate the position of a planet at a given time.
    
    Parameters:
    -----------
    planet_name : str
        Name of the planet
    time : datetime or float
        Either a datetime object or days since J2000.0
    use_time : bool
        Whether to use time-based calculation or static position
        
    Returns:
    --------
    tuple
        (x, y, z) position in AU (ecliptic coordinates)
    """
    if planet_name not in orbital_params_time:
        raise ValueError(f"Unknown planet: {planet_name}")
    
    params = orbital_params_time[planet_name]
    a, e, i, Ω, ω, M0, n = params
    
    if use_time:
        # Convert time to days since J2000.0
        if isinstance(time, datetime):
            jd = julian_date(time)
            days_since_j2000 = jd - 2451545.0
        else:
            days_since_j2000 = time
        
        # Calculate mean anomaly at given time
        M = np.radians(M0 + n * days_since_j2000)
        
        # Solve Kepler's equation for eccentric anomaly
        E = solve_kepler_equation(M, e)
        
        # Calculate true anomaly
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), 
                           np.sqrt(1 - e) * np.cos(E/2))
    else:
        # Use static position based on original implementation
        planet_index = list(orbital_params_time.keys()).index(planet_name)
        nu = np.radians((planet_index * 30) % 360)
    
    # Convert angles to radians
    i, Ω, ω = np.radians([i, Ω, ω])
    
    # Calculate position in orbital plane
    r = a * (1 - e**2) / (1 + e * np.cos(nu))
    x_orbit = r * np.cos(nu)
    y_orbit = r * np.sin(nu)
    
    # Apply argument of perihelion
    x_temp = x_orbit * np.cos(ω) - y_orbit * np.sin(ω)
    y_temp = x_orbit * np.sin(ω) + y_orbit * np.cos(ω)
    
    # Apply inclination
    y_incl = y_temp * np.cos(i)
    z_incl = y_temp * np.sin(i)
    
    # Apply longitude of ascending node
    x = x_temp * np.cos(Ω) - y_incl * np.sin(Ω)
    y = x_temp * np.sin(Ω) + y_incl * np.cos(Ω)
    z = z_incl
    
    return x, y, z


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
    'Neptune': [30.05, 0.0113, 1.77,     131.783,    273.187,   0.035],
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
    'Oort Cloud': {'inner_radius': 2000, 'outer_radius': 100000, 'color': 'rgba(200,200,255,0.3)', 'opacity': 0.05},
    'Termination Shock': {'inner_radius': 117, 'outer_radius': 120, 'tailward_offset': 32, 'north_distance': 27, 'port_side': 12,'color': 'rgba(255,100,100,0.3)', 'opacity': 0.2}, 
    'Heliosheath': {'inner_edge': 80, 'outer_edge': 100, 'windward_thickness': 90, 'color': 'rgba(255,100,100,0.3)', 'opacity': 0.2}
    # Bow shock and heliotail have a more complicated implementation seen below at another point. 
}

# New on first flight ***********

# Main Moon orbital parameters by planet. 
# Earth
moons = {
    'earth_moon': [384748, 385000, 363300, 405507, 0.0549006, 6.687, 5.15, 1.543, 27.322, 29.530],
    'mars_phobos': [9376, 11.08 ,9234.42, 9517.58, 0.0151, 0.0, 26.04, 1.093, .31891023, 0.0 ], 
    'mars_deimos': [23463.2, 6.27, 23455.5, 23470.9, 0.00033, 0.0, 27.58, 0.93, 1.263, 0.0], 
        # Jupiter
    'jupiter_io': [421800, 1821.3, 420000, 423400, 0.0041, 0.0, 0.04, 0.0, 1.769, 0.0],
    'jupiter_europa': [671100, 1560.8, 664862, 677338, 0.009, 0.1, 0.47, 0.0, 3.551, 0.0],
    'jupiter_ganymede': [1070400, 2634.1, 1069200, 1071600, 0.0013, 0.33, 0.21, 0.0, 7.155, 0.0],
    'jupiter_callisto': [1882700, 2410.3, 1865800, 1899600, 0.0074, 0.0, 0.51, 0.0, 16.69, 0.0],
    
    # Saturn
    'saturn_titan': [1221870, 2574.7, 1186680, 1257060, 0.0288, 0.0, 0.33, 0.0, 15.945, 0.0],
    
    # Uranus
    'uranus_ariel': [191020, 578.9, 190900, 191140, 0.0012, 0.0, 0.04, 0.0, 2.520, 0.0],
    'uranus_umbriel': [266000, 584.7, 265800, 266200, 0.0039, 0.0, 0.13, 0.0, 4.144, 0.0],
    'uranus_titania': [435910, 788.9, 435730, 436090, 0.0011, 0.0, 0.08, 0.0, 8.706, 0.0],
    'uranus_oberon': [583520, 761.4, 583000, 584000, 0.0014, 0.0, 0.07, 0.0, 13.46, 0.0],
    'uranus_miranda': [129390, 235.8, 129370, 129410, 0.0013, 0.0, 4.34, 0.0, 1.413, 0.0],
    
    # Neptune
    'neptune_triton': [354759, 1353.4, 354759, 354759, 0.000016, 0.0, 157.35, 0.0, 5.877, 0.0],
    
    # Pluto
    'pluto_charon': [19596, 606.0, 19596, 19596, 0.0, 0.0, 0.0, 0.0, 6.387, 0.0]

} 

# Will need to fix the colors to what actually makes sense. 
moon_colors = {
    'Moon': '#8c8680', 'Phobos': '#e6c89c', 'Deimos': '#4f71be', 'Io': '#d1603d',
    'Europa': '#e0ae6f', 'Ganymede': '#c5ab6e', 'Callisto': '#9fc4e7', 'Titan': '#4f71be',
    'Ariel': '#8c8680', 'Umbriel': '#ab9c8a', 'Titania': '#d9d9d9', 'Oberon': '#d9d9d9',
    'Miranda': '#c49e6c', 'Triton': '#bb5540', 'Charon': '#bb6a50'
}

# Relative planetary radii (Earth = 1), will need to calculate
moon_relative_sizes = {
    'Moon': 0.383, 'Phobos': 0.950, 'Deimos': 1.0, 'Io': 0.532,
    'Europa': 11.21, 'Ganymede': 9.45, 'Callisto': 4.01, 'Titan': 3.88,
    'Ariel': 0.074, 'Umbriel': 0.186, 'Titania': 0.183, 'Oberon': 0.16,
    'Miranda': 0.18, 'Triton': 0.10, 'Charon': 0.12
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

# Time based Calcs 



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
    

def create_bow_shock_ellipsoid(standoff_distance=130, base_radius=180, length=1500, 
                             width_factor=0.85, height_factor=0.85, 
                             color='rgba(255,150,100,0.25)', opacity=0.2, 
                             points=250, use_galactic=False):
    """
    Create a bow shock as an ellipsoid.
    
    Parameters:
    - standoff_distance: Distance from sun to bow shock nose (AU)
    - base_radius: Base radius of the ellipsoid (AU)
    - length: Length of the ellipsoid (AU)
    - width_factor: Width factor relative to base_radius
    - height_factor: Height factor relative to base_radius
    - color: Color of the bow shock
    - opacity: Opacity of the bow shock
    - points: Resolution of the mesh
    - use_galactic: Whether to convert to galactic coordinates
    
    Returns:
    - A Plotly Surface trace object
    """
    # Create parametric surface for a full ellipsoid
    u = np.linspace(0, 2*np.pi, points)
    v = np.linspace(0, np.pi, points)
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Calculate the center position (shifted forward from origin)
    center_x = standoff_distance - length/2
    
    # Semi-axes of the ellipsoid
    a = length/2
    b = base_radius * width_factor
    c = base_radius * height_factor
    
    # Calculate coordinates of the ellipsoid
    x = center_x + a * np.cos(u_grid) * np.sin(v_grid)
    y = b * np.sin(u_grid) * np.sin(v_grid)
    z = c * np.cos(v_grid)
    
    # Convert to galactic coordinates if requested
    if use_galactic:
        x, y, z = galactic_converter.ecliptic_to_galactic(x, y, z)
    
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, color]],
        opacity=opacity, 
        showscale=False,
        name='Bow Shock',
        hoverinfo='name',
        lighting=dict(
            ambient=0.6,
            diffuse=0.7,
            specular=0.1,
            roughness=0.9,
            fresnel=0.1
        )
    )

# Helper: create a deformed ellipsoid for the termination shock
def create_termination_shock(inner_radius=117, outer_radius=120, tailward_offset=32, 
                            north_distance=27, port_side=12, color='rgba(255,100,100,0.3)', 
                            opacity=0.2, points=50, use_galactic=False):
    """
    Create a deformed ellipsoid to represent the heliosphere's termination shock.
    
    Parameters:
    - inner_radius: Base radius for the shock in the nose direction (AU)
    - outer_radius: Extended radius for the shock in the tail direction (AU)
    - tailward_offset: Extra distance in the tail direction (AU)
    - north_distance: Extra extension in the north direction (AU)
    - port_side: Extra extension on the port side (AU)
    - color: Color of the termination shock
    - opacity: Opacity of the termination shock
    - points: Resolution of the mesh
    - use_galactic: Whether to convert to galactic coordinates
    
    Returns:
    - A Plotly Surface trace object
    """
    # Create parametric surface meshgrid
    u = np.linspace(0, 2*np.pi, points)
    v = np.linspace(0, np.pi, points)
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Base ellipsoid parameters
    a = outer_radius  # Semi-major axis in tail direction (X)
    b = inner_radius  # Semi-minor axis in side directions (Y)
    c = inner_radius  # Semi-minor axis in polar directions (Z)
    
    # Apply deformations for the asymmetric shape
    # X direction (Sun-tail line)
    x_base = a * np.cos(u_grid) * np.sin(v_grid)
    # Apply tail deformation (elongate in -X direction)
    x = np.where(x_base < 0, 
                 x_base * (1 + tailward_offset/a), 
                 x_base * 0.8)  # Compress slightly in nose direction
    
    # Y direction (port-starboard)
    y_base = b * np.sin(u_grid) * np.sin(v_grid)
    # Apply port side deformation
    y = np.where(y_base < 0,
                y_base * (1 + port_side/b),
                y_base)
    
    # Z direction (north-south)
    z_base = c * np.cos(v_grid)
    # Apply north deformation
    z = np.where(z_base > 0,
                z_base * (1 + north_distance/c),
                z_base * 0.9)  # Slightly flatten the south side
    
    # Convert to galactic coordinates if requested
    if use_galactic:
        x, y, z = galactic_converter.ecliptic_to_galactic(x, y, z)
    
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, color]],
        opacity=opacity, 
        showscale=False,
        name='Termination Shock',
        hoverinfo='name',
        lighting=dict(
            ambient=0.7,
            diffuse=0.8,
            specular=0.2,
            roughness=0.7,
            fresnel=0.2
        )
    )

# Also add a heliotail function to create the extended tail feature
def create_heliotail(base_radius=120, length=1000, width_factor=0.6, height_factor=0.5,
                   color='rgba(200,100,100,0.2)', opacity=0.15, points=40, use_galactic=False):
    """
    Create an elongated tail structure extending from the termination shock.
    
    Parameters:
    - base_radius: Radius at the base of the tail (AU)
    - length: Length of the tail extending backward (AU)
    - width_factor, height_factor: Factors to determine tail width and height
    - color: Color of the heliotail
    - opacity: Opacity of the heliotail
    - points: Resolution of the tail mesh
    - use_galactic: Whether to convert to galactic coordinates
    
    Returns:
    - A Plotly Surface trace object
    """
    # Create parametric cone surface
    u = np.linspace(0, 2*np.pi, points)
    v = np.linspace(0, 1, points//2)  # v parameter goes from base to tip
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Calculate tapering radius
    radius = base_radius * (1 - v_grid*0.8)  # Taper to 20% of original size
    
    # Calculate coordinates with elongation in -X direction
    x = -v_grid * length
    y = radius * width_factor * np.cos(u_grid)
    z = radius * height_factor * np.sin(u_grid)
    
    # Convert to galactic coordinates if requested
    if use_galactic:
        x, y, z = galactic_converter.ecliptic_to_galactic(x, y, z)
    
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, color]],
        opacity=opacity, 
        showscale=False,
        name='Heliotail',
        hoverinfo='name',
        lighting=dict(
            ambient=0.6,
            diffuse=0.7,
            specular=0.1,
            roughness=0.8,
            fresnel=0.1
        )
    )

# Additional function for bow shock ahead of heliosphere
def create_bow_shock(standoff_distance=40, radius=130, thickness=15, 
                   color='rgba(255,150,100,0.25)', opacity=0.2, points=50, use_galactic=False):
    """
    Create a bow shock surface ahead of the heliosphere.
    
    Parameters:
    - standoff_distance: Distance from sun to bow shock nose (AU)
    - radius: Radius of the bow shock at its widest (AU)
    - thickness: Thickness of the bow shock layer (AU)
    - color: Color of the bow shock
    - opacity: Opacity of the bow shock
    - points: Resolution of the bow shock mesh
    - use_galactic: Whether to convert to galactic coordinates
    
    Returns:
    - A Plotly Surface trace object
    """
    # Create parametric surface for a partial ellipsoid (front half only)
    u = np.linspace(-np.pi/2, np.pi/2, points)  # Only front half
    v = np.linspace(0, np.pi, points//2)
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Calculate coordinates of a paraboloid
    # We use a coefficient to make it more parabolic
    parabola_coef = 0.008
    x = standoff_distance + thickness * np.cos(u_grid) * np.sin(v_grid)
    y_base = radius * np.sin(u_grid) * np.sin(v_grid)
    z_base = radius * np.cos(v_grid)
    
    # Apply parabolic deformation
    x_squared = np.maximum(0, x - standoff_distance)**2
    y = y_base + parabola_coef * x_squared * np.sign(y_base)
    z = z_base + parabola_coef * x_squared * np.sign(z_base)
    
    # Convert to galactic coordinates if requested
    if use_galactic:
        x, y, z = galactic_converter.ecliptic_to_galactic(x, y, z)
    
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, color]],
        opacity=opacity, 
        showscale=False,
        name='Bow Shock',
        hoverinfo='name',
        lighting=dict(
            ambient=0.6,
            diffuse=0.7,
            specular=0.1,
            roughness=0.9,
            fresnel=0.1
        )
    )

# Update the make_traces function to include these new components
def make_traces_with_heliosphere(planet_list, include_regions=True, use_planet_spheres=True, 
                              include_ellipsoid=False, ellipsoid_params=None, use_galactic=False,
                              show_termination_shock=True, show_heliotail=True, show_bow_shock=True):
    """
    Enhanced version of make_traces that includes heliosphere boundary features
    """
    # Get basic traces from original function
    traces = make_traces(planet_list, include_regions, use_planet_spheres, 
                       include_ellipsoid, ellipsoid_params, use_galactic)
    
    # Add termination shock if requested
    if show_termination_shock:
        ts_params = solar_regions['Termination Shock']
        ts_trace = create_termination_shock(
            ts_params['inner_radius'], ts_params['outer_radius'],
            ts_params['tailward_offset'], ts_params['north_distance'],
            ts_params['port_side'], ts_params['color'], ts_params['opacity'],
            use_galactic=use_galactic
        )
        traces.append(ts_trace)
    
    # Add heliotail if requested
    if show_heliotail:
        # Use termination shock parameters to match dimensions
        ts_params = solar_regions['Termination Shock']
        tail_trace = create_heliotail(
            base_radius=ts_params['outer_radius'],
            length=800,  # Reasonable length for visualization
            use_galactic=use_galactic
        )
        traces.append(tail_trace)
    
    # Add bow shock if requested
    if show_bow_shock:
        bow_trace = create_bow_shock(use_galactic=use_galactic)
        traces.append(bow_trace)
    
    return traces


# Enhanced build_fig function that uses the new traces
def build_heliosphere_fig(subset, rng, title, include_regions=True, include_oort=False, 
                        use_planet_spheres=True, include_ellipsoid=False, ellipsoid_params=None, 
                        use_galactic=False, show_termination_shock=True, 
                        show_heliotail=True, show_bow_shock=True):
    
    traces = make_traces_with_heliosphere(
        subset, include_regions, use_planet_spheres, 
        include_ellipsoid, ellipsoid_params, use_galactic,
        show_termination_shock, show_heliotail, show_bow_shock
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

# Example function to showcase the heliosphere
def heliosphere_demo(view_range=200):
    """
    Create a visualization of the heliosphere including termination shock, heliotail, and bow shock
    
    Parameters:
    -----------
    view_range : float
        Range for the visualization in AU
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figure with the heliosphere visualization
    """
    # Use outer planets for scale reference
    planets = ['Jupiter', 'Saturn', 'Uranus', 'Neptune']
    
    # Create the figure with heliosphere components
    fig = build_heliosphere_fig(
        planets, view_range, 
        "Solar System Heliosphere",
        include_regions=True,
        use_planet_spheres=True,
        show_termination_shock=True,
        show_heliotail=True,
        show_bow_shock=True
    )
    
    # Add annotations to highlight components
    fig.update_layout(
        scene=dict(
            annotations=[
                dict(
                    x=100, y=0, z=0,
                    text="Termination Shock",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='red'
                ),
                dict(
                    x=-300, y=0, z=0,
                    text="Heliotail",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='darkred'
                ),
                dict(
                    x=130, y=0, z=0,
                    text="Bow Shock",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='orange'
                )
            ]
        )
    )
    
    return fig




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

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Update the solar_regions dictionary to add missing heliosheath parameters
solar_regions.update({
    'Termination Shock': {
        'inner_radius': 80,    # Inner radius of termination shock (AU)
        'outer_radius': 100,   # Outer radius of termination shock (AU)
        'tailward_offset': 80, # Tailward extension (more stretched in -X direction)
        'north_distance': 20,  # North pole extension (more stretched in +Z)
        'port_side': 15,       # Port side extension (more stretched in -Y)
        'color': 'rgba(255,100,100,0.3)',
        'opacity': 0.3
    },
    'Bow Shock': {
        'standoff_distance': 130,  # Distance from sun to bow shock nose (AU)
        'base_radius': 180,
        'length': 1500,             # Radius of bow shock at widest (AU)           # Thickness of bow shock layer (AU)
        'width_factor': 0.8,
        'height_factor': 0.5,           
        'color': 'rgba(255,150,100,0.25)',
        'opacity': 0.2
    },
    'Heliotail': {
        'base_radius': 100,    # Base radius where tail starts (AU)
        'length': 800,         # Length of tail extending backward (AU)
        'width_factor': 0.6,   # Width relative to base radius
        'height_factor': 0.5,  # Height relative to base radius
        'color': 'rgba(200,100,100,0.2)',
        'opacity': 0.15
    }
})

# Example 1: Create a simple demonstration of the termination shock
def demo_termination_shock():
    # Use outer planets as reference points
    planets = ['Jupiter', 'Saturn', 'Uranus', 'Neptune']
    
    # Create figure with termination shock
    fig = build_heliosphere_fig(
        planets, 
        rng=200,  # View range in AU
        title="Solar System with Termination Shock",
        show_termination_shock=True,
        show_heliotail=False,
        show_bow_shock=False
    )
    
    return fig

# Example 2: Full heliosphere visualization with all components
def full_heliosphere_visualization():
    # Include outer planets for scale reference
    planets = ['Jupiter', 'Saturn', 'Uranus', 'Neptune']
    
    # Create comprehensive visualization
    fig = build_heliosphere_fig(
        planets,
        rng=400,  # Larger view to see entire heliotail
        title="Complete Heliosphere Visualization",
        show_termination_shock=True,
        show_heliotail=True,
        show_bow_shock=True
    )
    
    # Add text annotations to highlight key features
    fig.update_layout(
        scene=dict(
            annotations=[
                dict(
                    x=100, y=0, z=30,
                    text="Termination Shock",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='red'
                ),
                dict(
                    x=-300, y=0, z=0,
                    text="Heliotail",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='darkred'
                ),
                dict(
                    x=150, y=50, z=0,
                    text="Bow Shock",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='orange'
                ),
                dict(
                    x=30, y=0, z=0,
                    text="Solar Wind",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='blue'
                )
            ]
        )
    )
    
    return fig

# Example 3: Side-by-side comparison of ecliptic vs galactic coordinates
def heliosphere_coordinate_comparison():
    planets = ['Jupiter', 'Saturn', 'Uranus', 'Neptune']
    view_range = 250
    
    # Create figures in both coordinate systems
    fig_ecliptic = build_heliosphere_fig(
        planets, view_range, "Heliosphere (Ecliptic Coordinates)",
        show_termination_shock=True, show_heliotail=True, show_bow_shock=True,
        use_galactic=False
    )
    
    fig_galactic = build_heliosphere_fig(
        planets, view_range, "Heliosphere (Galactic Coordinates)",
        show_termination_shock=True, show_heliotail=True, show_bow_shock=True, 
        use_galactic=True
    )
    
    # Create subplot with both views
    subplot_fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Ecliptic View", "Galactic View")
    )
    
    # Add traces from individual figures
    for trace in fig_ecliptic.data:
        subplot_fig.add_trace(trace, row=1, col=1)
    
    for trace in fig_galactic.data:
        subplot_fig.add_trace(trace, row=1, col=2)
    
    # Update layout
    subplot_fig.update_layout(
        title="Heliosphere: Ecliptic vs Galactic Coordinate Systems",
        width=1400,
        height=700
    )
    
    return subplot_fig

# Example function to run
def run_examples():
    # Select which example to display
    # return demo_termination_shock()
    return full_heliosphere_visualization()
    # return heliosphere_coordinate_comparison()

# This would be called in a Jupyter notebook
# run_examples()

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

# New on first flight ***********
# Helper: Create a sphere for the moons selected (included atm) in the moons dictionary above. 
def create_moon_sphere(x, y, z, radius, color, name, resolution=10, use_galactic=False):
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
        
        # # Heliopause (spherical boundary)
        # hp_params = solar_regions['Heliopause']
        # hp_trace = create_shell(hp_params['radius'], None, hp_params['color'], 
        #                      hp_params['opacity'], use_galactic=use_galactic)
        # hp_trace.name = 'Heliopause'
        # traces.append(hp_trace)

        # Termination Shock (spherical shell)
        # ts_params = solar_regions['Termination Shock']
        # ts_trace = create_shell(ts_params['inner_radius'], ts_params['outer_radius'],
        #                         ts_params['tailward_offset'], ts_params['north_distance'],
        #                         ts_params['port_side'], ts_params['color'], ts_params['opacity'], use_galactic=use_galactic)
        # ts_trace.name = 'Termination Shock'
        # traces.append(ts_trace)

    
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

# Time Based Calcs and fig generation

def make_traces_time(planet_list, current_time=None, show_trails=True, trail_days=365, 
                    include_regions=True, use_planet_spheres=True, 
                    include_ellipsoid=False, ellipsoid_params=None, use_galactic=False):
    """
    Enhanced version of make_traces that includes time-based planet positions.
    
    Parameters:
    -----------
    planet_list : list
        List of planet names
    current_time : datetime or None
        Time for planet positions (None uses J2000.0)
    show_trails : bool
        Whether to show orbital trails
    trail_days : float
        Number of days to show in trails
    include_regions : bool
        Whether to include solar system regions
    use_planet_spheres : bool
        Whether to use 3D spheres for planets
    include_ellipsoid : bool
        Whether to include custom ellipsoid
    ellipsoid_params : dict
        Parameters for custom ellipsoid
    use_galactic : bool
        Whether to use galactic coordinates
    
    Returns:
    --------
    list
        List of Plotly traces
    """
    traces = []
    
    if current_time is None:
        current_time = datetime(2000, 1, 1, 12, 0, 0)  # J2000.0 epoch
    
    # Sun sphere (unchanged)
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x_s = 0.05 * np.cos(u) * np.sin(v)
    y_s = 0.05 * np.sin(u) * np.sin(v)
    z_s = 0.05 * np.cos(v)
    
    if use_galactic:
        x_s, y_s, z_s = galactic_converter.ecliptic_to_galactic(x_s, y_s, z_s)
    
    traces.append(go.Surface(
        x=x_s, y=y_s, z=z_s,
        colorscale=[[0,'yellow'],[1,'yellow']],
        opacity=0.9, 
        showscale=False,
        name="Sun",
        lighting=dict(ambient=0.9, diffuse=0.8, specular=0.5)
    ))
    
    # Add regions if requested (unchanged)
    if include_regions:
        ab_params = solar_regions['Asteroid Belt']
        traces.append(create_disk(ab_params['inner_radius'], ab_params['outer_radius'], 
                                ab_params['color'], ab_params['opacity'], use_galactic=use_galactic))
        
        kb_params = solar_regions['Kuiper Belt']
        kb_trace = create_shell(kb_params['inner_radius'], kb_params['outer_radius'], 
                              kb_params['color'], kb_params['opacity'], use_galactic=use_galactic)
        kb_trace.name = 'Kuiper Belt'
        traces.append(kb_trace)
    
    # Add custom ellipsoid if requested (unchanged)
    if include_ellipsoid and ellipsoid_params:
        traces.append(create_ellipsoid(**ellipsoid_params, use_galactic=use_galactic))
    
    # Planets with time-based positions
    for planet_name in planet_list:
        # Get current position
        x, y, z = calculate_orbital_position(planet_name, current_time)
        
        # Convert to galactic if needed
        if use_galactic:
            x, y, z = galactic_converter.ecliptic_to_galactic(x, y, z)
        
        # Create planet representation
        if use_planet_spheres:
            planet_radius = planet_relative_sizes[planet_name] * size_scale_factor
            planet_trace = create_planet_sphere(
                x, y, z, planet_radius, 
                planet_colors[planet_name], planet_name,
                use_galactic=False  # Already converted if needed
            )
            traces.append(planet_trace)
        else:
            traces.append(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers', 
                marker=dict(size=8, color=planet_colors[planet_name]), 
                name=planet_name
            ))
        
        # Add orbital trails if requested
        if show_trails:
            trail_points = 50
            trail_times = np.linspace(-trail_days/2, trail_days/2, trail_points)
            trail_x, trail_y, trail_z = [], [], []
            
            for dt in trail_times:
                trail_time = current_time + timedelta(days=dt)
                tx, ty, tz = calculate_orbital_position(planet_name, trail_time)
                
                if use_galactic:
                    tx, ty, tz = galactic_converter.ecliptic_to_galactic(tx, ty, tz)
                
                trail_x.append(tx)
                trail_y.append(ty)
                trail_z.append(tz)
            
            # Add trail
            opacity = 0.3 if dt < 0 else 0.6  # Dimmer for past, brighter for future
            traces.append(go.Scatter3d(
                x=trail_x, y=trail_y, z=trail_z,
                mode='lines',
                line=dict(color=planet_colors[planet_name], width=2),
                opacity=opacity,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    return traces

def build_fig_time(subset, rng, title, current_time=None, show_trails=True, 
                  trail_days=365, include_regions=True, include_oort=False, 
                  use_planet_spheres=True, include_ellipsoid=False, 
                  ellipsoid_params=None, use_galactic=False):
    """
    Enhanced version of build_fig that includes time functionality.
    """
    traces = make_traces_time(
        subset, current_time, show_trails, trail_days,
        include_regions, use_planet_spheres, 
        include_ellipsoid, ellipsoid_params, use_galactic
    )
    
    # Add Oort Cloud if requested
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
        title = f"{title} (Galactic Coordinates)"
    else:
        x_label = 'X (AU)'
        y_label = 'Y (AU)'
        z_label = 'Z (AU)'
    
    # Add time information to title
    if current_time:
        title += f" - {current_time.strftime('%Y-%m-%d')}"
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-rng,rng], title=x_label),
            yaxis=dict(range=[-rng,rng], title=y_label),
            zaxis=dict(range=[-rng,rng], title=z_label),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        title=title,
        height=700,
        width=700,
        autosize=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def create_time_animation(planet_list, start_date, end_date, frames=30, rng=10,
                        include_regions=True, use_galactic=False):
    """
    Create an animated visualization of the solar system over time.
    
    Parameters:
    -----------
    planet_list : list
        List of planet names
    start_date : datetime
        Starting date for animation
    end_date : datetime
        Ending date for animation
    frames : int
        Number of frames in animation
    rng : float
        Range for visualization axes
    include_regions : bool
        Whether to include solar system regions
    use_galactic : bool
        Whether to use galactic coordinates
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Animated figure
    """
    # Create time steps
    time_steps = []
    total_days = (end_date - start_date).days
    for i in range(frames):
        time_steps.append(start_date + timedelta(days=i * total_days / (frames - 1)))
    
    # Create initial figure
    fig = build_fig_time(planet_list, rng, "Solar System Animation", 
                        current_time=start_date, show_trails=False,
                        include_regions=include_regions, use_galactic=use_galactic)
    
    # Create frames
    frames_list = []
    for i, time in enumerate(time_steps):
        frame_data = []
        
        # Update planet positions
        for j, planet_name in enumerate(planet_list):
            x, y, z = calculate_orbital_position(planet_name, time)
            
            if use_galactic:
                x, y, z = galactic_converter.ecliptic_to_galactic(x, y, z)
            
            # Update planet sphere or marker position
            if use_planet_spheres:
                planet_radius = planet_relative_sizes[planet_name] * size_scale_factor
                u, v = np.mgrid[0:2*np.pi:20*1j, 0:np.pi:10*1j]
                sphere_x = x + planet_radius * np.cos(u) * np.sin(v)
                sphere_y = y + planet_radius * np.sin(u) * np.sin(v)
                sphere_z = z + planet_radius * np.cos(v)
                
                frame_data.append(go.Surface(
                    x=sphere_x, y=sphere_y, z=sphere_z,
                    colorscale=[[0, planet_colors[planet_name]], 
                              [1, planet_colors[planet_name]]],
                    showscale=False,
                    name=planet_name
                ))
            else:
                frame_data.append(go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode='markers',
                    marker=dict(size=8, color=planet_colors[planet_name]),
                    name=planet_name
                ))
        
        frames_list.append(go.Frame(
            data=frame_data,
            name=str(i),
            layout=dict(title=f"Solar System - {time.strftime('%Y-%m-%d')}")
        ))
    
    fig.frames = frames_list
    
    # Add animation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(label='Play',
                         method='animate',
                         args=[None, {'frame': {'duration': 100, 'redraw': True},
                                     'fromcurrent': True}]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                       'mode': 'immediate',
                                       'transition': {'duration': 0}}])
                ]
            )
        ],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 16},
                'prefix': 'Date: ',
                'visible': True,
                'xanchor': 'right'
            },
            'steps': [
                {'args': [[str(i)], 
                         {'frame': {'duration': 100, 'redraw': True},
                          'mode': 'immediate',
                          'transition': {'duration': 0}}],
                 'label': time_steps[i].strftime('%Y-%m-%d'),
                 'method': 'animate'}
                for i in range(frames)
            ]
        }]
    )
    
    return fig

# Example usage functions
def demo_time_based_positions():
    """
    Demo showing planets at different times
    """
    # Current date
    current_date = datetime.now()
    
    # Create visualization for current date
    planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter']
    fig = build_fig_time(
        planets, 10, "Inner Solar System", 
        current_time=current_date, 
        show_trails=True, 
        trail_days=90
    )
    
    return fig

def demo_planetary_alignment(date=None):
    """
    Show all planets at a specific date to check for alignments
    """
    if date is None:
        date = datetime(2024, 1, 1)
    
    all_planets = list(orbital_params_time.keys())[:8]  # Main planets only
    
    fig = build_fig_time(
        all_planets, 35, f"Planetary Positions - {date.strftime('%Y-%m-%d')}", 
        current_time=date, 
        show_trails=False
    )
    
    return fig

def demo_animation():
    """
    Create an animation of planetary motion
    """
    start = datetime(2024, 1, 1)
    end = datetime(2025, 1, 1)
    
    fig = create_time_animation(
        ['Mercury', 'Venus', 'Earth', 'Mars'],
        start, end, frames=24, rng=2
    )
    
    return fig

# Additional time-related utility functions
def find_planetary_conjunctions(planet1, planet2, start_date, end_date, threshold=5.0):
    """
    Find when two planets are in conjunction (appear close together in the sky).
    
    Parameters:
    -----------
    planet1, planet2 : str
        Names of the planets
    start_date, end_date : datetime
        Date range to search
    threshold : float
        Maximum angular separation in degrees to consider conjunction
    
    Returns:
    --------
    list
        List of (date, separation) tuples for conjunctions
    """
    conjunctions = []
    
    # Check every day in the range
    current_date = start_date
    while current_date <= end_date:
        # Get positions
        x1, y1, z1 = calculate_orbital_position(planet1, current_date)
        x2, y2, z2 = calculate_orbital_position(planet2, current_date)
        
        # Calculate angular separation as seen from origin (Sun)
        r1 = np.sqrt(x1**2 + y1**2 + z1**2)
        r2 = np.sqrt(x2**2 + y2**2 + z2**2)
        
        dot_product = (x1*x2 + y1*y2 + z1*z2) / (r1 * r2)
        separation = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))
        
        if separation <= threshold:
            conjunctions.append((current_date, separation))
        
        current_date += timedelta(days=1)
    
    return conjunctions

def get_planet_distances_from_earth(date):
    """
    Calculate distances of all planets from Earth at a given date.
    
    Parameters:
    -----------
    date : datetime
        The date for calculation
    
    Returns:
    --------
    dict
        Dictionary of planet names to distances in AU
    """
    # Get Earth's position
    earth_x, earth_y, earth_z = calculate_orbital_position('Earth', date)
    
    distances = {}
    for planet in orbital_params_time.keys():
        if planet == 'Earth':
            continue
            
        # Get planet position
        x, y, z = calculate_orbital_position(planet, date)
        
        # Calculate distance
        distance = np.sqrt((x - earth_x)**2 + (y - earth_y)**2 + (z - earth_z)**2)
        distances[planet] = distance
    
    return distances

def calculate_synodic_period(planet1, planet2):
    """
    Calculate the synodic period between two planets.
    
    Parameters:
    -----------
    planet1, planet2 : str
        Names of the planets
    
    Returns:
    --------
    float
        Synodic period in days
    """
    n1 = orbital_params_time[planet1][6]  # Mean motion in deg/day
    n2 = orbital_params_time[planet2][6]
    
    # Synodic period formula
    synodic = 360.0 / abs(n1 - n2)
    
    return synodic