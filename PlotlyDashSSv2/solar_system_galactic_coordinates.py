import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

class GalacticCoordinates:
    """
    A class to handle coordinate transformations between Solar System and Galactic coordinates.
    
    The Galactic coordinate system has:
    - Origin at the Solar System barycenter
    - X-axis pointing toward the Galactic center (l = 0°, b = 0°)
    - Y-axis pointing in the direction of Galactic rotation (l = 90°, b = 0°)
    - Z-axis pointing toward the North Galactic Pole (b = 90°)
    
    Where:
    - l is the Galactic longitude
    - b is the Galactic latitude
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
        # Obliquity of the ecliptic (J2000)
        epsilon = np.radians(23.4392911)
        
        # Transformation
        x_eq = x
        y_eq = y * np.cos(epsilon) - z * np.sin(epsilon)
        z_eq = y * np.sin(epsilon) + z * np.cos(epsilon)
        
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
    
    def spherical_to_cartesian(self, r, longitude, latitude):
        """
        Convert from spherical Galactic coordinates to cartesian.
        
        Parameters:
        -----------
        r : float or array
            Distance from the origin (AU)
        longitude : float or array
            Galactic longitude (radians)
        latitude : float or array
            Galactic latitude (radians)
        
        Returns:
        --------
        x, y, z : float or array
            Cartesian coordinates (AU)
        """
        x = r * np.cos(latitude) * np.cos(longitude)
        y = r * np.cos(latitude) * np.sin(longitude)
        z = r * np.sin(latitude)
        return x, y, z
    
    def cartesian_to_spherical(self, x, y, z):
        """
        Convert from cartesian to spherical Galactic coordinates.
        
        Parameters:
        -----------
        x, y, z : float or array
            Cartesian coordinates (AU)
        
        Returns:
        --------
        r : float or array
            Distance from the origin (AU)
        longitude : float or array
            Galactic longitude (radians)
        latitude : float or array
            Galactic latitude (radians)
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        longitude = np.arctan2(y, x)
        latitude = np.arcsin(z / r)
        return r, longitude, latitude


class SolarSystem:
    """A basic solar system model with planets and their orbits."""
    
    def __init__(self):
        self.planets = {
            'Mercury': {'a': 0.387, 'e': 0.206, 'i': np.radians(7.0), 'color': 'gray'},
            'Venus': {'a': 0.723, 'e': 0.007, 'i': np.radians(3.4), 'color': 'orange'},
            'Earth': {'a': 1.0, 'e': 0.017, 'i': np.radians(0.0), 'color': 'blue'},
            'Mars': {'a': 1.524, 'e': 0.093, 'i': np.radians(1.9), 'color': 'red'},
            'Jupiter': {'a': 5.203, 'e': 0.048, 'i': np.radians(1.3), 'color': 'brown'},
            'Saturn': {'a': 9.537, 'e': 0.056, 'i': np.radians(2.5), 'color': 'gold'},
            'Uranus': {'a': 19.191, 'e': 0.046, 'i': np.radians(0.8), 'color': 'cyan'},
            'Neptune': {'a': 30.069, 'e': 0.010, 'i': np.radians(1.8), 'color': 'blue'},
        }
        
        # Randomly assign orbital phases
        for planet in self.planets.values():
            planet['phase'] = np.random.uniform(0, 2 * np.pi)
        
        # Create coordinate transformer
        self.galactic = GalacticCoordinates()
    
    def calculate_positions(self, time):
        """Calculate positions of all planets at a given time."""
        positions = {}
        
        for name, planet in self.planets.items():
            a = planet['a']  # Semi-major axis
            e = planet['e']  # Eccentricity
            i = planet['i']  # Inclination
            phase = planet['phase']  # Orbital phase
            
            # Calculate orbital position (simplified model)
            # This approximates orbits as circular for simplicity
            angle = time * (1.0 / a**1.5) + phase
            
            # Orbital position in ecliptic plane
            r = a * (1 - e**2) / (1 + e * np.cos(angle))
            x_ecl = r * np.cos(angle)
            y_ecl = r * np.sin(angle) * np.cos(i)
            z_ecl = r * np.sin(angle) * np.sin(i)
            
            # Convert to Galactic coordinates
            x_gal, y_gal, z_gal = self.galactic.ecliptic_to_galactic(x_ecl, y_ecl, z_ecl)
            
            positions[name] = {
                'ecliptic': (x_ecl, y_ecl, z_ecl),
                'galactic': (x_gal, y_gal, z_gal),
                'color': planet['color']
            }
        
        return positions

    def visualize(self, time_span=50, steps=100):
        """Visualize the solar system in both coordinate systems."""
        fig = plt.figure(figsize=(15, 7))
        
        # Set up ecliptic view
        ax_ecliptic = fig.add_subplot(121, projection='3d')
        ax_ecliptic.set_title('Solar System (Ecliptic Coordinates)')
        ax_ecliptic.set_xlabel('X (AU)')
        ax_ecliptic.set_ylabel('Y (AU)')
        ax_ecliptic.set_zlabel('Z (AU)')
        
        # Set up galactic view
        ax_galactic = fig.add_subplot(122, projection='3d')
        ax_galactic.set_title('Solar System (Galactic Coordinates)')
        ax_galactic.set_xlabel('X (Galactic Center Direction, AU)')
        ax_galactic.set_ylabel('Y (Galactic Rotation Direction, AU)')
        ax_galactic.set_zlabel('Z (North Galactic Pole, AU)')
        
        # Add Sun
        ax_ecliptic.scatter([0], [0], [0], color='yellow', s=100, label='Sun')
        ax_galactic.scatter([0], [0], [0], color='yellow', s=100, label='Sun')
        
        # Add coordinate system indicator for Galactic
        arrow_length = 2.0
        ax_galactic.quiver(0, 0, 0, arrow_length, 0, 0, color='red', label='Galactic Center')
        ax_galactic.quiver(0, 0, 0, 0, arrow_length, 0, color='green', label='Galactic Rotation')
        ax_galactic.quiver(0, 0, 0, 0, 0, arrow_length, color='blue', label='Galactic North')
        
        # Create planets with trajectories
        planet_points = {}
        trajectories = {}
        
        # Generate orbit data
        for t in np.linspace(0, time_span, steps):
            positions = self.calculate_positions(t)
            
            for name, pos in positions.items():
                if name not in trajectories:
                    trajectories[name] = {'ecliptic': [], 'galactic': []}
                
                trajectories[name]['ecliptic'].append(pos['ecliptic'])
                trajectories[name]['galactic'].append(pos['galactic'])
        
        # Plot trajectories
        for name, traj in trajectories.items():
            x_ecl, y_ecl, z_ecl = zip(*traj['ecliptic'])
            x_gal, y_gal, z_gal = zip(*traj['galactic'])
            
            color = self.planets[name]['color']
            
            ax_ecliptic.plot(x_ecl, y_ecl, z_ecl, color=color, alpha=0.5)
            ax_galactic.plot(x_gal, y_gal, z_gal, color=color, alpha=0.5)
            
            # Plot the starting position
            ax_ecliptic.scatter([x_ecl[0]], [y_ecl[0]], [z_ecl[0]], color=color, s=50, label=name)
            ax_galactic.scatter([x_gal[0]], [y_gal[0]], [z_gal[0]], color=color, s=50, label=name)
        
        # Add legends
        ax_ecliptic.legend()
        ax_galactic.legend()
        
        # Set equal aspect ratio
        max_range = 35
        ax_ecliptic.set_xlim(-max_range, max_range)
        ax_ecliptic.set_ylim(-max_range, max_range)
        ax_ecliptic.set_zlim(-max_range, max_range)
        
        ax_galactic.set_xlim(-max_range, max_range)
        ax_galactic.set_ylim(-max_range, max_range)
        ax_galactic.set_zlim(-max_range, max_range)
        
        plt.tight_layout()
        plt.show()
    
    def animate(self, time_span=20, steps=200):
        """Create an animation of the solar system in both coordinate systems."""
        fig = plt.figure(figsize=(15, 7))
        
        # Set up ecliptic view
        ax_ecliptic = fig.add_subplot(121, projection='3d')
        ax_ecliptic.set_title('Solar System (Ecliptic Coordinates)')
        ax_ecliptic.set_xlabel('X (AU)')
        ax_ecliptic.set_ylabel('Y (AU)')
        ax_ecliptic.set_zlabel('Z (AU)')
        
        # Set up galactic view
        ax_galactic = fig.add_subplot(122, projection='3d')
        ax_galactic.set_title('Solar System (Galactic Coordinates)')
        ax_galactic.set_xlabel('X (Galactic Center Direction, AU)')
        ax_galactic.set_ylabel('Y (Galactic Rotation Direction, AU)')
        ax_galactic.set_zlabel('Z (North Galactic Pole, AU)')
        
        # Add Sun
        ax_ecliptic.scatter([0], [0], [0], color='yellow', s=100)
        ax_galactic.scatter([0], [0], [0], color='yellow', s=100)
        
        # Add coordinate system indicator for Galactic
        arrow_length = 2.0
        ax_galactic.quiver(0, 0, 0, arrow_length, 0, 0, color='red')
        ax_galactic.quiver(0, 0, 0, 0, arrow_length, 0, color='green')
        ax_galactic.quiver(0, 0, 0, 0, 0, arrow_length, color='blue')
        
        # Set view limits
        max_range = 35
        ax_ecliptic.set_xlim(-max_range, max_range)
        ax_ecliptic.set_ylim(-max_range, max_range)
        ax_ecliptic.set_zlim(-max_range, max_range)
        
        ax_galactic.set_xlim(-max_range, max_range)
        ax_galactic.set_ylim(-max_range, max_range)
        ax_galactic.set_zlim(-max_range, max_range)
        
        # Create scatter points for planets
        planet_scatter_ecl = {}
        planet_scatter_gal = {}
        
        for name, planet in self.planets.items():
            color = planet['color']
            planet_scatter_ecl[name] = ax_ecliptic.scatter([], [], [], color=color, s=30, label=name)
            planet_scatter_gal[name] = ax_galactic.scatter([], [], [], color=color, s=30)
        
        # Add legends
        ax_ecliptic.legend()
        
        def init():
            for scatter in planet_scatter_ecl.values():
                scatter._offsets3d = (np.array([]), np.array([]), np.array([]))
            for scatter in planet_scatter_gal.values():
                scatter._offsets3d = (np.array([]), np.array([]), np.array([]))
            return []
        
        def animate(i):
            t = i * time_span / steps
            positions = self.calculate_positions(t)
            
            for name, pos in positions.items():
                x_ecl, y_ecl, z_ecl = pos['ecliptic']
                x_gal, y_gal, z_gal = pos['galactic']
                
                planet_scatter_ecl[name]._offsets3d = ([x_ecl], [y_ecl], [z_ecl])
                planet_scatter_gal[name]._offsets3d = ([x_gal], [y_gal], [z_gal])
            
            return []
        
        anim = FuncAnimation(fig, animate, init_func=init, frames=steps, interval=50, blit=True)
        plt.tight_layout()
        plt.show()
        
        return anim


# Example usage
if __name__ == "__main__":
    # Create a solar system instance
    solar_system = SolarSystem()
    
    # Visualize the static representation
    print("Generating static visualization...")
    solar_system.visualize()
    
    # Create an animation (uncomment to run)
    print("Generating animation...")
    anim = solar_system.animate()
    
    # Example of coordinate conversion
    print("\nCoordinate Conversion Example:")
    
    # Define a point in ecliptic coordinates (AU)
    x_ecl, y_ecl, z_ecl = 1.0, 0.0, 0.0  # 1 AU along the X-axis in ecliptic coordinates
    
    # Convert to Galactic coordinates
    galactic = GalacticCoordinates()
    x_gal, y_gal, z_gal = galactic.ecliptic_to_galactic(x_ecl, y_ecl, z_ecl)
    
    print(f"Ecliptic coordinates: ({x_ecl:.3f}, {y_ecl:.3f}, {z_ecl:.3f}) AU")
    print(f"Galactic coordinates: ({x_gal:.3f}, {y_gal:.3f}, {z_gal:.3f}) AU")
    
    # Convert back to check accuracy
    x_ecl_back, y_ecl_back, z_ecl_back = galactic.galactic_to_equatorial(*galactic.equatorial_to_galactic(*galactic.ecliptic_to_equatorial(x_ecl, y_ecl, z_ecl)))
    print(f"Converted back to ecliptic: ({x_ecl_back:.3f}, {y_ecl_back:.3f}, {z_ecl_back:.3f}) AU")
