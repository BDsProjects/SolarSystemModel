# pages/time_explorer.py
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
sys.path.append('..')  # Add parent directory to path
from solar_helpers_galactic import (
    build_fig_time, create_time_animation, 
    get_planet_distances_from_earth, find_planetary_conjunctions,
    orbital_params_time, calculate_synodic_period
)

dash.register_page(__name__, path='/time-explorer', name='Time Explorer')

# Define available planets
all_planets = list(orbital_params_time.keys())
main_planets = all_planets[:8]  # Mercury through Neptune

layout = html.Div([
    html.H2("Solar System Time Explorer", style={'textAlign': 'center'}),
    
    # Main controls
    html.Div([
        # Date and time picker
        html.Div([
            html.H3("Select Date & Time:"),
            dcc.DatePickerSingle(
                id='time-date-picker',
                date=datetime.now().date(),
                display_format='YYYY-MM-DD',
                style={'marginBottom': '10px'}
            ),
            html.Div([
                html.Label("Hour: "),
                dcc.Input(
                    id='hour-input',
                    type='number',
                    value=12,
                    min=0,
                    max=23,
                    style={'width': '50px'}
                ),
                html.Label(" : ", style={'margin': '0 5px'}),
                dcc.Input(
                    id='minute-input',
                    type='number',
                    value=0,
                    min=0,
                    max=59,
                    style={'width': '50px'}
                ),
                html.Label(" UTC", style={'marginLeft': '10px'})
            ], style={'marginTop': '10px'}),
            html.Button('Now', id='now-button', style={'marginTop': '10px'})
        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
        
        # Planet selection
        html.Div([
            html.H3("Select Planets:"),
            dcc.Dropdown(
                id='time-planet-dropdown',
                options=[{'label': planet, 'value': planet} for planet in all_planets],
                value=['Mercury', 'Venus', 'Earth', 'Mars'],
                multi=True,
                style={'width': '100%'}
            ),
            html.Button('Main Planets', id='main-planets-btn', style={'margin': '5px'}),
            html.Button('Outer Objects', id='outer-objects-btn', style={'margin': '5px'})
        ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
        
        # Visualization options
        html.Div([
            html.H3("Visualization Options:"),
            dcc.Checklist(
                id='time-options-checklist',
                options=[
                    {'label': 'Show Orbital Trails', 'value': 'trails'},
                    {'label': 'Show Regions', 'value': 'regions'},
                    {'label': 'Use Galactic Coordinates', 'value': 'galactic'},
                    {'label': '3D Planet Spheres', 'value': 'spheres'}
                ],
                value=['trails', 'regions', 'spheres'],
                inline=False
            ),
            html.Div([
                html.Label("Trail Length (days):"),
                dcc.Slider(
                    id='time-trail-slider',
                    min=7,
                    max=365,
                    step=7,
                    value=30,
                    marks={i: f'{i}d' for i in [7, 30, 90, 180, 365]}
                )
            ], style={'marginTop': '10px'}),
            html.Div([
                html.Label("View Range (AU):"),
                dcc.Input(
                    id='time-range-input',
                    type='number',
                    value=5,
                    min=1,
                    max=100,
                    style={'width': '80px'}
                )
            ], style={'marginTop': '10px'})
        ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
    ]),
    
    # Main visualization
    dcc.Graph(id='time-solar-system-plot', style={'height': '700px', 'marginTop': '20px'}),
    
    # Information panels
    html.Div([
        # Distance information
        html.Div([
            html.H3("Distances from Earth"),
            html.Div(id='time-distance-info')
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
        
        # Special events
        html.Div([
            html.H3("Astronomical Events"),
            html.Div([
                html.Label("Check for conjunctions between:"),
                dcc.Dropdown(
                    id='planet1-dropdown',
                    options=[{'label': p, 'value': p} for p in main_planets],
                    value='Venus',
                    style={'width': '150px', 'display': 'inline-block', 'margin': '5px'}
                ),
                html.Span(" and ", style={'margin': '0 10px'}),
                dcc.Dropdown(
                    id='planet2-dropdown',
                    options=[{'label': p, 'value': p} for p in main_planets],
                    value='Mars',
                    style={'width': '150px', 'display': 'inline-block', 'margin': '5px'}
                ),
                html.Button('Find Conjunctions', id='find-conjunctions-btn', 
                           style={'margin': '10px'})
            ]),
            html.Div(id='conjunction-results')
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
    ]),
    
    # Time controls
    html.Div([
        html.H3("Time Navigation", style={'textAlign': 'center'}),
        html.Div([
            html.Button('« Year', id='prev-year-btn', style={'margin': '5px'}),
            html.Button('« Month', id='prev-month-btn', style={'margin': '5px'}),
            html.Button('« Day', id='prev-day-btn', style={'margin': '5px'}),
            html.Button('Today', id='today-btn', style={'margin': '5px'}),
            html.Button('Day »', id='next-day-btn', style={'margin': '5px'}),
            html.Button('Month »', id='next-month-btn', style={'margin': '5px'}),
            html.Button('Year »', id='next-year-btn', style={'margin': '5px'})
        ], style={'textAlign': 'center', 'marginTop': '20px'})
    ]),
    
    # Animation section
    html.Div([
        html.H3("Create Time-Lapse Animation"),
        html.Div([
            html.Div([
                html.Label("Start Date:"),
                dcc.DatePickerSingle(
                    id='anim-start-date',
                    date=(datetime.now() - timedelta(days=180)).date(),
                    display_format='YYYY-MM-DD'
                )
            ], style={'display': 'inline-block', 'margin': '10px'}),
            html.Div([
                html.Label("End Date:"),
                dcc.DatePickerSingle(
                    id='anim-end-date',
                    date=(datetime.now() + timedelta(days=180)).date(),
                    display_format='YYYY-MM-DD'
                )
            ], style={'display': 'inline-block', 'margin': '10px'}),
            html.Div([
                html.Label("Frames:"),
                dcc.Input(
                    id='anim-frames',
                    type='number',
                    value=24,
                    min=10,
                    max=100,
                    style={'width': '60px'}
                )
            ], style={'display': 'inline-block', 'margin': '10px'}),
            html.Button('Create Animation', id='create-time-animation-btn', 
                       style={'margin': '10px'})
        ], style={'textAlign': 'center'}),
        dcc.Loading(
            id="loading-animation",
            type="default",
            children=[dcc.Graph(id='time-animation-plot', style={'height': '600px'})]
        )
    ], style={'marginTop': '40px', 'border': '1px solid #ddd', 'padding': '20px'})
])

# Callbacks
@callback(
    Output('time-solar-system-plot', 'figure'),
    [Input('time-date-picker', 'date'),
     Input('hour-input', 'value'),
     Input('minute-input', 'value'),
     Input('time-planet-dropdown', 'value'),
     Input('time-options-checklist', 'value'),
     Input('time-trail-slider', 'value'),
     Input('time-range-input', 'value')]
)
def update_time_plot(selected_date, hour, minute, selected_planets, options, trail_days, view_range):
    """Update the main plot based on user selections."""
    if not selected_planets:
        return go.Figure().add_annotation(text="Please select at least one planet", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Parse datetime
    try:
        date = datetime.strptime(selected_date, '%Y-%m-%d')
        date = date.replace(hour=hour, minute=minute)
    except:
        date = datetime.now()
    
    # Parse options
    show_trails = 'trails' in options
    show_regions = 'regions' in options
    use_galactic = 'galactic' in options
    use_spheres = 'spheres' in options
    
    # Create figure
    fig = build_fig_time(
        selected_planets,
        view_range,
        "Solar System",
        current_time=date,
        show_trails=show_trails,
        trail_days=trail_days,
        include_regions=show_regions,
        use_planet_spheres=use_spheres,
        use_galactic=use_galactic
    )
    
    return fig

@callback(
    Output('time-distance-info', 'children'),
    Input('time-date-picker', 'date'),
    Input('hour-input', 'value'),
    Input('minute-input', 'value')
)
def update_distance_info(selected_date, hour, minute):
    """Update the distance information table."""
    try:
        date = datetime.strptime(selected_date, '%Y-%m-%d')
        date = date.replace(hour=hour, minute=minute)
    except:
        date = datetime.now()
    
    distances = get_planet_distances_from_earth(date)
    
    # Create table
    rows = []
    rows.append(html.Tr([html.Th("Planet"), html.Th("Distance (AU)")]))
    
    for planet, distance in sorted(distances.items(), key=lambda x: x[1]):
        if planet in main_planets:  # Only show main planets
            rows.append(html.Tr([
                html.Td(planet),
                html.Td(f"{distance:.3f}")
            ]))
    
    return html.Table(rows, style={'width': '100%'})

@callback(
    Output('conjunction-results', 'children'),
    Input('find-conjunctions-btn', 'n_clicks'),
    [State('planet1-dropdown', 'value'),
     State('planet2-dropdown', 'value'),
     State('time-date-picker', 'date')]
)
def find_conjunctions_callback(n_clicks, planet1, planet2, current_date):
    """Find conjunctions between two planets."""
    if n_clicks is None:
        return ""
    
    try:
        date = datetime.strptime(current_date, '%Y-%m-%d')
    except:
        date = datetime.now()
    
    # Search for conjunctions in the next year
    start_date = date
    end_date = date + timedelta(days=365)
    
    conjunctions = find_planetary_conjunctions(planet1, planet2, start_date, end_date, threshold=5.0)
    
    if not conjunctions:
        return html.P(f"No conjunctions between {planet1} and {planet2} in the next year.")
    
    # Create results
    results = [html.P(f"Conjunctions between {planet1} and {planet2}:")]
    for conj_date, separation in conjunctions:
        results.append(html.Li(f"{conj_date.strftime('%Y-%m-%d')}: {separation:.2f}° separation"))
    
    # Add synodic period information
    synodic = calculate_synodic_period(planet1, planet2)
    results.append(html.P(f"Average time between conjunctions: {synodic:.1f} days", 
                         style={'marginTop': '10px', 'fontStyle': 'italic'}))
    
    return html.Div(results)

@callback(
    Output('time-date-picker', 'date'),
    Output('hour-input', 'value'),
    Output('minute-input', 'value'),
    [Input('now-button', 'n_clicks'),
     Input('prev-year-btn', 'n_clicks'),
     Input('prev-month-btn', 'n_clicks'),
     Input('prev-day-btn', 'n_clicks'),
     Input('today-btn', 'n_clicks'),
     Input('next-day-btn', 'n_clicks'),
     Input('next-month-btn', 'n_clicks'),
     Input('next-year-btn', 'n_clicks')],
    [State('time-date-picker', 'date'),
     State('hour-input', 'value'),
     State('minute-input', 'value')]
)
def update_datetime(now_clicks, py_clicks, pm_clicks, pd_clicks, 
                   today_clicks, nd_clicks, nm_clicks, ny_clicks,
                   current_date, current_hour, current_minute):
    """Handle time navigation buttons."""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return current_date, current_hour, current_minute
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        date = datetime.strptime(current_date, '%Y-%m-%d')
        date = date.replace(hour=current_hour, minute=current_minute)
    except:
        date = datetime.now()
    
    if button_id == 'now-button':
        date = datetime.now()
    elif button_id == 'today-btn':
        date = datetime.now().replace(hour=12, minute=0)
    elif button_id == 'prev-year-btn':
        date = date.replace(year=date.year - 1)
    elif button_id == 'next-year-btn':
        date = date.replace(year=date.year + 1)
    elif button_id == 'prev-month-btn':
        if date.month == 1:
            date = date.replace(year=date.year - 1, month=12)
        else:
            date = date.replace(month=date.month - 1)
    elif button_id == 'next-month-btn':
        if date.month == 12:
            date = date.replace(year=date.year + 1, month=1)
        else:
            date = date.replace(month=date.month + 1)
    elif button_id == 'prev-day-btn':
        date = date - timedelta(days=1)
    elif button_id == 'next-day-btn':
        date = date + timedelta(days=1)
    
    return date.strftime('%Y-%m-%d'), date.hour, date.minute

@callback(
    Output('time-planet-dropdown', 'value'),
    [Input('main-planets-btn', 'n_clicks'),
     Input('outer-objects-btn', 'n_clicks')]
)
def update_planet_selection(main_clicks, outer_clicks):
    """Update planet selection based on preset buttons."""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'main-planets-btn':
        return main_planets
    elif button_id == 'outer-objects-btn':
        return ['Pluto', 'Eris', 'Makemake', 'Haumea', 'Sedna']
    
    return dash.no_update

@callback(
    Output('time-animation-plot', 'figure'),
    Input('create-time-animation-btn', 'n_clicks'),
    [State('anim-start-date', 'date'),
     State('anim-end-date', 'date'),
     State('anim-frames', 'value'),
     State('time-planet-dropdown', 'value'),
     State('time-options-checklist', 'value'),
     State('time-range-input', 'value')]
)
def create_animation_callback(n_clicks, start_date, end_date, frames, 
                             selected_planets, options, view_range):
    """Create time-lapse animation."""
    if n_clicks is None or not selected_planets:
        return go.Figure()
    
    # Parse dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Parse options
    show_regions = 'regions' in options
    use_galactic = 'galactic' in options
    
    # Create animation
    fig = create_time_animation(
        selected_planets,
        start,
        end,
        frames=frames,
        rng=view_range,
        include_regions=show_regions,
        use_galactic=use_galactic,
        use_planet_spheres=True  # Add this parameter
    )
    
    return fig