"""
Verify the planetary positions database structure and contents
"""
import sqlite3
import pandas as pd

def verify_database(db_filename='planetary_positions.db'):
    """Verify the database structure and print summary information."""
    
    try:
        conn = sqlite3.connect(db_filename)
        cursor = conn.cursor()
        
        print(f"Successfully connected to {db_filename}\n")
        
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='planetary_positions'
        """)
        
        if not cursor.fetchone():
            print("ERROR: planetary_positions table does not exist!")
            return
        
        print("✓ Table 'planetary_positions' exists\n")
        
        # Get table schema
        cursor.execute("PRAGMA table_info(planetary_positions)")
        columns = cursor.fetchall()
        
        print("Table Schema:")
        print("-" * 50)
        for col in columns:
            print(f"Column: {col[1]:12} Type: {col[2]:10} Not Null: {col[3]} Default: {col[4]}")
        print("-" * 50 + "\n")
        
        # Check for index
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name='idx_planet_date'
        """)
        
        if cursor.fetchone():
            print("✓ Index 'idx_planet_date' exists\n")
        else:
            print("⚠ Index 'idx_planet_date' does not exist\n")
        
        # Get summary statistics
        cursor.execute("SELECT COUNT(*) FROM planetary_positions")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT planet_name) FROM planetary_positions")
        total_planets = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(date), MAX(date) FROM planetary_positions")
        date_range = cursor.fetchone()
        
        cursor.execute("SELECT DISTINCT planet_name FROM planetary_positions ORDER BY planet_name")
        planet_names = [row[0] for row in cursor.fetchall()]
        
        print("Database Statistics:")
        print("-" * 50)
        print(f"Total records: {total_records:,}")
        print(f"Total planets: {total_planets}")
        print(f"Date range: {date_range[0]} to {date_range[1]}")
        print(f"Planets: {', '.join(planet_names)}")
        print("-" * 50 + "\n")
        
        # Sample data
        print("Sample data (first 5 records):")
        df = pd.read_sql_query("SELECT * FROM planetary_positions LIMIT 5", conn)
        print(df)
        
        # Check for specific date
        sample_date = '2024-01-01'
        cursor.execute("""
            SELECT COUNT(*) FROM planetary_positions 
            WHERE date = ?
        """, (sample_date,))
        count_on_date = cursor.fetchone()[0]
        
        print(f"\nRecords for {sample_date}: {count_on_date}")
        
        # Display Earth's position on this date as example
        cursor.execute("""
            SELECT * FROM planetary_positions 
            WHERE planet_name = 'Earth' AND date = ?
        """, (sample_date,))
        
        earth_data = cursor.fetchone()
        if earth_data:
            print(f"\nEarth's position on {sample_date}:")
            print(f"  Ecliptic: ({earth_data[4]:.6f}, {earth_data[5]:.6f}, {earth_data[6]:.6f}) AU")
            print(f"  Galactic: ({earth_data[7]:.6f}, {earth_data[8]:.6f}, {earth_data[9]:.6f}) AU")
        
        conn.close()
        print("\n✓ Database verification complete!")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_database()