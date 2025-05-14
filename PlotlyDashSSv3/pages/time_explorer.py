# pages/time_explorer.py
import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import sys
sys.path.append('.')
from solar_helpers_galactic import (
    build_fig_time, create_time_animation,
    find_planetary_conjunctions, calculate_synodic_period,
    get_planet_distances_from_earth, orbital_params_time
)

dash.register_page(__name__)

# Define available planets
all_planets = list(orbital_params_time.keys())[:8]  # Main planets only

layout = html.Div([
    html.H1("Time-Based Solar System Explorer", style={'textAlign': 'center'}),
    
    # Main tabs
    dcc.Tabs(id='time-tabs', value='current-view', children=[
        dcc.Tab(label='Current View', value='current-view'),
        dcc.Tab(label='Historical Events', value='historical'),
        dcc.Tab(label='Conjunctions', value='conjunctions'),
        dcc.Tab(label='Orbital Periods', value='periods'),
    ]),
    
    html.Div(id='time-tab-content', style={'marginTop': '20px'})
])

@callback(
    Output('time-tab-content', 'children'),
    [Input('time-tabs', 'value')]
)
def render_time_tab(active_tab):
    if active_tab == 'current-view':
        return html.Div([
            html.Div([
                html.H3("Select Date and Time:"),
                dcc.DatePickerSingle(
                    id='datetime-picker',
                    date=datetime.now().date(),
                    display_format='YYYY-MM-DD'
                ),
                dcc.Input(
                    id='time-input',
                    type='text',
                    placeholder='HH:MM',
                    value='12:00',
                    style={'marginLeft': '10px', 'width': '80px'}
                ),
                html.Button('Update', id='update-time-btn', 
                           style={'marginLeft': '10px'})
            ]),
            
            html.Div([
                html.H3("Select Planets:"),
                dcc.Checklist(
                    id='time-planet-checklist',
                    options=[{'label': planet, 'value': planet} 
                            for planet in all_planets],
                    value=['Mercury', 'Venus', 'Earth', 'Mars'],
                    inline=True
                )
            ], style={'marginTop': '20px'}),
            
            html.Div([
                html.H3("Visualization Options:"),
                dcc.RadioButtons(
                    id='coord-system',
                    options=[
                        {'label': 'Ecliptic Coordinates', 'value': 'ecliptic'},
                        {'label': 'Galactic Coordinates', 'value': 'galactic'}
                    ],
                    value='ecliptic',
                    inline=True
                ),
                dcc.Checklist(
                    id='view-options',
                    options=[
                        {'label': 'Show Orbital Trails', 'value': 'trails'},
                        {'label': 'Show Solar System Regions', 'value': 'regions'},
                        {'label': 'Show Past Positions', 'value': 'past'},
                        {'label': 'Show Future Positions', 'value': 'future'}
                    ],
                    value=['trails', 'regions'],
                    inline=True,
                    style={'marginTop': '10px'}
                )
            ], style={'marginTop': '20px'}),
            
            dcc.Graph(id='time-based-plot', style={'height': '700px'}),
            
            html.Div(id='planet-info', style={'marginTop': '20px'})
        ])
    
    elif active_tab == 'historical':
        return html.Div([
            html.H3("Historical Planetary Positions"),
            html.P("View planetary configurations at significant historical dates"),
            
            dcc.Dropdown(
                id='historical-events',
                options=[
                    {'label': 'Apollo 11 Moon Landing (July 20, 1969)', 
                     'value': '1969-07-20'},
                    {'label': 'Voyager 1 Launch (September 5, 1977)', 
                     'value': '1977-09-05'},
                    {'label': 'Halley\'s Comet Perihelion (February 9, 1986)', 
                     'value': '1986-02-09'},
                    {'label': 'Pluto Discovery (February 18, 1930)', 
                     'value': '1930-02-18'},
                    {'label': 'New Horizons Pluto Flyby (July 14, 2015)', 
                     'value': '2015-07-14'},
                    {'label': 'Great Conjunction 2020 (December 21, 2020)', 
                     'value': '2020-12-21'},
                    {'label': 'Y2K (January 1, 2000)', 
                     'value': '2000-01-01'},
                    {'label': 'Custom Date', 'value': 'custom'}
                ],
                value='2020-12-21'
            ),
            
            html.Div(id='custom-date-input', style={'marginTop': '10px'}),
            
            dcc.Graph(id='historical-plot', style={'height': '700px'}),
            
            html.Div(id='historical-info', style={'marginTop': '20px'})
        ])
    
    elif active_tab == 'conjunctions':
        return html.Div([
            html.H3("Planetary Conjunctions"),
            html.P("Find when planets appear close together in the sky"),
            
            html.Div([
                html.Label("Planet 1:"),
                dcc.Dropdown(
                    id='planet1-dropdown',
                    options=[{'label': p, 'value': p} for p in all_planets],
                    value='Jupiter',
                    style={'width': '200px'}
                ),
                html.Label("Planet 2:", style={'marginLeft': '20px'}),
                dcc.Dropdown(
                    id='planet2-dropdown',
                    options=[{'label': p, 'value': p} for p in all_planets],
                    value='Saturn',
                    style={'width': '200px'}
                )
            ], style={'display': 'flex', 'alignItems': 'center'}),
            
            html.Div([
                html.Label("Date Range:"),
                dcc.DatePickerRange(
                    id='conjunction-date-range',
                    start_date=(datetime.now() - timedelta(days=365)).date(),
                    end_date=(datetime.now() + timedelta(days=365)).date(),
                    display_format='YYYY-MM-DD'
                ),
                html.Button('Find Conjunctions', id='find-conjunctions-btn',
                           style={'marginLeft': '20px'})
            ], style={'marginTop': '20px'}),
            
            html.Div(id='conjunction-results', style={'marginTop': '20px'}),
            
            dcc.Graph(id='conjunction-plot', style={'height': '500px'})
        ])
    
    elif active_tab == 'periods':
        return html.Div([
            html.H3("Orbital Periods and Resonances"),
            html.P("Explore planetary orbital periods and their relationships"),
            
            html.Div(id='period-table'),
            
            html.H4("Synodic Periods", style={'marginTop': '30px'}),
            html.P("Time between successive oppositions or conjunctions as seen from Earth"),
            
            html.Div(id='synodic-table'),
            
            html.H4("Orbital Resonances", style={'marginTop': '30px'}),
            dcc.Graph(id='resonance-plot', style={'height': '500px'})
        ])

# Callbacks for current view tab
@callback(
    [Output('time-based-plot', 'figure'),
     Output('planet-info', 'children')],
    [Input('update-time-btn', 'n_clicks')],
    [State('datetime-picker', 'date'),
     State('time-input', 'value'),
     State('time-planet-checklist', 'value'),
     State('coord-system', 'value'),
     State('view-options', 'value')]
)
def update_time_based_view(n_clicks, date_str, time_str, planets, coord_system, options):
    # Parse datetime
    try:
        hours, minutes = map(int, time_str.split(':'))
        date = datetime.strptime(date_str, '%Y-%m-%d')
        date = date.replace(hour=hours, minute=minutes)
    except:
        date = datetime.now()
    
    # Create figure
    fig = build_fig_time(
        planets,
        rng=10 if 'Mercury' in planets else 40,
        title=f"Solar System on {date.strftime('%Y-%m-%d %H:%M')}",
        current_time=date,
        show_trails='trails' in options,
        trail_days=90,
        include_regions='regions' in options,
        use_galactic=coord_system == 'galactic'
    )
    
    # Calculate planet information
    distances = get_planet_distances_from_earth(date)
    info_rows = []
    
    for planet in planets:
        if planet == 'Earth':
            continue
        distance = distances.get(planet, 0)
        info_rows.append(
            html.Tr([
                html.Td(planet),
                html.Td(f"{distance:.3f} AU"),
                html.Td(f"{distance * 149.597871:.1f} million km")
            ])
        )
    
    info_table = html.Table([
        html.Thead([
            html.Tr([
                html.Th("Planet"),
                html.Th("Distance from Earth"),
                html.Th("Distance (million km)")
            ])
        ]),
        html.Tbody(info_rows)
    ], style={'margin': 'auto', 'width': '50%'})
    
    return fig, info_table

# Callbacks for historical tab
@callback(
    Output('custom-date-input', 'children'),
    [Input('historical-events', 'value')]
)
def show_custom_date_input(event_value):
    if event_value == 'custom':
        return html.Div([
            dcc.DatePickerSingle(
                id='custom-historical-date',
                date=datetime.now().date(),
                display_format='YYYY-MM-DD'
            )
        ])
    return html.Div()

@callback(
    [Output('historical-plot', 'figure'),
     Output('historical-info', 'children')],
    [Input('historical-events', 'value'),
     Input('custom-historical-date', 'date')]
)
def update_historical_view(event_value, custom_date):
    if event_value == 'custom' and custom_date:
        date_str = custom_date
        event_name = "Custom Date"
    else:
        date_str = event_value if event_value != 'custom' else '2000-01-01'
        event_name = next((opt['label'] for opt in 
                          [{'label': 'Apollo 11 Moon Landing (July 20, 1969)', 
                            'value': '1969-07-20'},
                           # ... other options
                          ] if opt['value'] == event_value), "Unknown Event")
    
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Create figure for all planets
    fig = build_fig_time(
        all_planets,
        rng=40,
        title=f"{event_name} - Planetary Positions",
        current_time=date,
        show_trails=False,
        include_regions=True
    )
    
    # Create information about the configuration
    info = html.Div([
        html.H4(f"Planetary Configuration on {date.strftime('%B %d, %Y')}"),
        html.P(f"Event: {event_name}"),
        # Add more specific information about the event if needed
    ])
    
    return fig, info

# Callbacks for conjunctions tab
@callback(
    [Output('conjunction-results', 'children'),
     Output('conjunction-plot', 'figure')],
    [Input('find-conjunctions-btn', 'n_clicks')],
    [State('planet1-dropdown', 'value'),
     State('planet2-dropdown', 'value'),
     State('conjunction-date-range', 'start_date'),
     State('conjunction-date-range', 'end_date')]
)
def find_conjunctions_callback(n_clicks, planet1, planet2, start_date, end_date):
    if n_clicks is None:
        return html.Div(), {}
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Find conjunctions
    conjunctions = find_planetary_conjunctions(planet1, planet2, start, end)
    
    if not conjunctions:
        return html.P("No conjunctions found in the specified date range"), {}
    
    # Create results table
    table_rows = [
        html.Tr([
            html.Th("Date"),
            html.Th("Separation (degrees)")
        ])
    ]
    
    dates = []
    separations = []
    
    for date, separation in conjunctions:
        table_rows.append(
            html.Tr([
                html.Td(date.strftime('%Y-%m-%d')),
                html.Td(f"{separation:.2f}°")
            ])
        )
        dates.append(date)
        separations.append(separation)
    
    results = html.Div([
        html.H4(f"Conjunctions between {planet1} and {planet2}"),
        html.Table(table_rows, style={'margin': 'auto'})
    ])
    
    # Create plot
    fig = go.Figure(data=[
        go.Scatter(
            x=dates,
            y=separations,
            mode='lines+markers',
            name='Angular Separation'
        )
    ])
    
    fig.update_layout(
        title=f"Angular Separation: {planet1} - {planet2}",
        xaxis_title="Date",
        yaxis_title="Separation (degrees)",
        hovermode='x unified'
    )
    
    return results, fig

# Callbacks for periods tab
@callback(
    [Output('period-table', 'children'),
     Output('synodic-table', 'children'),
     Output('resonance-plot', 'figure')],
    [Input('time-tabs', 'value')]
)
def update_periods_tab(active_tab):
    if active_tab != 'periods':
        return html.Div(), html.Div(), {}
    
    # Create orbital period table
    period_rows = [
        html.Tr([
            html.Th("Planet"),
            html.Th("Orbital Period (days)"),
            html.Th("Orbital Period (years)"),
            html.Th("Mean Motion (°/day)")
        ])
    ]
    
    for planet, params in orbital_params_time.items():
        if planet in all_planets:
            n = params[6]  # Mean motion in degrees/day
            period_days = 360.0 / n
            period_years = period_days / 365.25
            
            period_rows.append(
                html.Tr([
                    html.Td(planet),
                    html.Td(f"{period_days:.1f}"),
                    html.Td(f"{period_years:.2f}"),
                    html.Td(f"{n:.4f}")
                ])
            )
    
    period_table = html.Table(period_rows, style={'margin': 'auto'})
    
    # Create synodic period table
    synodic_rows = [
        html.Tr([
            html.Th("Planet Pair"),
            html.Th("Synodic Period (days)"),
            html.Th("Synodic Period (years)")
        ])
    ]
    
    earth_idx = all_planets.index('Earth')
    for i, planet in enumerate(all_planets):
        if i != earth_idx:
            synodic = calculate_synodic_period('Earth', planet)
            synodic_years = synodic / 365.25
            
            synodic_rows.append(
                html.Tr([
                    html.Td(f"Earth - {planet}"),
                    html.Td(f"{synodic:.1f}"),
                    html.Td(f"{synodic_years:.2f}")
                ])
            )
    
    synodic_table = html.Table(synodic_rows, style={'margin': 'auto'})
    
    # Create resonance visualization
    periods = []
    planet_names = []
    
    for planet in all_planets:
        n = orbital_params_time[planet][6]
        period = 360.0 / n
        periods.append(period)
        planet_names.append(planet)
    
    fig = go.Figure(data=[
        go.Bar(
            x=planet_names,
            y=periods,
            name='Orbital Period (days)'
        )
    ])
    
    fig.update_layout(
        title="Planetary Orbital Periods",
        xaxis_title="Planet",
        yaxis_title="Period (days)",
        yaxis_type="log"
    )
    
    return period_table, synodic_table, fig