# FORMAT: [semi-major axis, mean radius, Perigee, Apogee, mean eccentricity, mean obliquity
# mean inclination of orbit to ecliptic, mean inclination of lunar equator to ecliptic,
#  period of orbit around earth (sidereal), period of orbit around earth (synodic)]
# in km and degrees for now, days for periods
# mercury, none

# Venus, none

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


