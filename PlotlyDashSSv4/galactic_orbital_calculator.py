"""
Calculate planetary positions in galactic coordinates from 1900 to 2500
and store in a SQL database.
"""
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

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
        
        # Pre-compute the obliquity of the ecliptic (J2000)
        self.epsilon = np.radians(23.4392911)
        
    def ecliptic_to_equatorial(self, x, y, z):
        """Convert from ecliptic to equatorial coordinates."""
        x_eq = x
        y_eq = y * np.cos(self.epsilon) - z * np.sin(self.epsilon)
        z_eq = y * np.sin(self.epsilon) + z * np.cos(self.epsilon)
        return x_eq, y_eq, z_eq
    
    def equatorial_to_galactic(self, x_eq, y_eq, z_eq):
        """Convert from equatorial to Galactic coordinates."""
        coords_eq = np.array([x_eq, y_eq, z_eq])
        coords_gal = self.eq_to_gal @ coords_eq
        return coords_gal[0], coords_gal[1], coords_gal[2]
    
    def ecliptic_to_galactic(self, x, y, z):
        """Convert directly from ecliptic to Galactic coordinates."""
        x_eq, y_eq, z_eq = self.ecliptic_to_equatorial(x, y, z)
        return self.equatorial_to_galactic(x_eq, y_eq, z_eq)


class OrbitalCalculator:
    """Calculate planetary positions based on orbital elements."""
    
    def __init__(self, orbital_params):
        self.orbital_params = orbital_params
        self.galactic_converter = GalacticCoordinates()
        
        # J2000.0 epoch (January 1, 2000, 12:00 TT)
        self.j2000_epoch = datetime(2000, 1, 1, 12, 0, 0)
        
    def julian_date(self, dt):
        """Convert datetime to Julian Date."""
        # Calculate Julian Day Number
        a = (14 - dt.month) // 12
        y = dt.year + 4800 - a
        m = dt.month + 12 * a - 3
        
        jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
        
        # Add fractional day
        jd = jdn + (dt.hour - 12) / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0
        
        return jd
    
    def days_since_j2000(self, dt):
        """Calculate days since J2000.0 epoch."""
        return (dt - self.j2000_epoch).total_seconds() / 86400.0
    
    def solve_kepler(self, M, e, tolerance=1e-8, max_iterations=50):
        """Solve Kepler's equation for eccentric anomaly."""
        # Initial guess for eccentric anomaly
        E = M if e < 0.8 else np.pi
        
        for _ in range(max_iterations):
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            E_new = E - f / f_prime
            
            if abs(E_new - E) < tolerance:
                return E_new
            E = E_new
            
        return E
    
    def orbital_position(self, planet_name, date):
        """Calculate the position of a planet at a given date."""
        # Get orbital parameters
        a, e, i, Omega, omega, M0, n = self.orbital_params[planet_name]
        
        # Convert angles to radians
        i_rad = np.radians(i)
        Omega_rad = np.radians(Omega)
        omega_rad = np.radians(omega)
        M0_rad = np.radians(M0)
        n_rad = np.radians(n)
        
        # Calculate days since J2000.0
        days = self.days_since_j2000(date)
        
        # Mean anomaly at the given date
        M = M0_rad + n_rad * days
        M = M % (2 * np.pi)  # Normalize to 0-2π
        
        # Solve Kepler's equation for eccentric anomaly
        E = self.solve_kepler(M, e)
        
        # Calculate true anomaly
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), 
                           np.sqrt(1 - e) * np.cos(E / 2))
        
        # Distance from focus
        r = a * (1 - e * np.cos(E))
        
        # Position in orbital plane
        x_orbital = r * np.cos(nu)
        y_orbital = r * np.sin(nu)
        z_orbital = 0
        
        # Rotate by argument of perihelion
        x_perihelion = x_orbital * np.cos(omega_rad) - y_orbital * np.sin(omega_rad)
        y_perihelion = x_orbital * np.sin(omega_rad) + y_orbital * np.cos(omega_rad)
        z_perihelion = z_orbital
        
        # Rotate by inclination
        x_inclined = x_perihelion
        y_inclined = y_perihelion * np.cos(i_rad) - z_perihelion * np.sin(i_rad)
        z_inclined = y_perihelion * np.sin(i_rad) + z_perihelion * np.cos(i_rad)
        
        # Rotate by longitude of ascending node
        x_ecliptic = x_inclined * np.cos(Omega_rad) - y_inclined * np.sin(Omega_rad)
        y_ecliptic = x_inclined * np.sin(Omega_rad) + y_inclined * np.cos(Omega_rad)
        z_ecliptic = z_inclined
        
        # Convert to galactic coordinates
        x_galactic, y_galactic, z_galactic = self.galactic_converter.ecliptic_to_galactic(
            x_ecliptic, y_ecliptic, z_ecliptic)
        
        return {
            'ecliptic': (x_ecliptic, y_ecliptic, z_ecliptic),
            'galactic': (x_galactic, y_galactic, z_galactic)
        }


def create_database():
    """Create the SQLite database and tables."""
    conn = sqlite3.connect('planetary_positions.db')
    cursor = conn.cursor()
    
    # Create table for planetary positions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS planetary_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        planet_name TEXT NOT NULL,
        date TEXT NOT NULL,
        julian_date REAL NOT NULL,
        x_ecliptic REAL NOT NULL,
        y_ecliptic REAL NOT NULL,
        z_ecliptic REAL NOT NULL,
        x_galactic REAL NOT NULL,
        y_galactic REAL NOT NULL,
        z_galactic REAL NOT NULL,
        UNIQUE(planet_name, date)
    )
    ''')
    
    # Create index for faster queries
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_planet_date 
    ON planetary_positions(planet_name, date)
    ''')
    
    conn.commit()
    return conn


def main():
    # Orbital parameters
    orbital_params_time = {
        # Name: [a(AU), e, i(deg), Ω(deg), ω(deg), M0(deg), n(deg/day)]
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
    
    # Create calculator
    calculator = OrbitalCalculator(orbital_params_time)
    
    # Create database
    conn = create_database()
    cursor = conn.cursor()
    
    # Generate dates from 1900 to 2500 (monthly intervals)
    start_date = datetime(1900, 1, 1)
    end_date = datetime(2500, 12, 31)
    
    # Calculate total number of months for progress bar
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    
    print(f"Calculating planetary positions from {start_date.year} to {end_date.year}")
    print(f"Total calculations: {total_months * len(orbital_params_time)}")
    
    # Batch size for database inserts
    batch_size = 1000
    batch_data = []
    
    # Calculate positions
    with tqdm(total=total_months * len(orbital_params_time)) as pbar:
        current_date = start_date
        
        while current_date <= end_date:
            for planet_name in orbital_params_time.keys():
                # Calculate position
                position = calculator.orbital_position(planet_name, current_date)
                
                # Prepare data for insertion
                data = (
                    planet_name,
                    current_date.strftime('%Y-%m-%d'),
                    calculator.julian_date(current_date),
                    position['ecliptic'][0],
                    position['ecliptic'][1],
                    position['ecliptic'][2],
                    position['galactic'][0],
                    position['galactic'][1],
                    position['galactic'][2]
                )
                
                batch_data.append(data)
                
                # Insert batch when full
                if len(batch_data) >= batch_size:
                    cursor.executemany('''
                    INSERT OR REPLACE INTO planetary_positions 
                    (planet_name, date, julian_date, x_ecliptic, y_ecliptic, z_ecliptic, 
                     x_galactic, y_galactic, z_galactic)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', batch_data)
                    conn.commit()
                    batch_data = []
                
                pbar.update(1)
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
    
    # Insert any remaining data
    if batch_data:
        cursor.executemany('''
        INSERT OR REPLACE INTO planetary_positions 
        (planet_name, date, julian_date, x_ecliptic, y_ecliptic, z_ecliptic, 
         x_galactic, y_galactic, z_galactic)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', batch_data)
        conn.commit()
    
    # Close database connection
    conn.close()
    
    print("\nDatabase creation complete!")
    print("Filename: planetary_positions.db")
    
    # Print some statistics
    conn = sqlite3.connect('planetary_positions.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM planetary_positions")
    total_records = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT planet_name) FROM planetary_positions")
    total_planets = cursor.fetchone()[0]
    
    cursor.execute("SELECT MIN(date), MAX(date) FROM planetary_positions")
    date_range = cursor.fetchone()
    
    print(f"\nDatabase statistics:")
    print(f"Total records: {total_records}")
    print(f"Total planets: {total_planets}")
    print(f"Date range: {date_range[0]} to {date_range[1]}")
    
    conn.close()


if __name__ == "__main__":
    main()