# pages/solar_system_info.py

from dash import html, dcc, register_page

register_page(__name__, path='/solar_system_info', name='Solar System Info')

layout = html.Div([
    html.H2("Solar System Information", className="mb-4"),
    
    html.Div([
        html.H3("Solar System Regions", className="mt-4 mb-3"),
        
        html.Div([
            html.H4("Asteroid Belt", className="text-primary"),
            html.P([
                "The asteroid belt is a torus-shaped region in the Solar System, located roughly between the orbits of the planets Mars and Jupiter. It contains a great many solid, irregularly shaped bodies, of many sizes but much smaller than planets, called asteroids or minor planets."
            ]),
            html.P([
                "• Distance from Sun: 2.2 to 3.2 AU",
                html.Br(),
                "• Estimated number of objects: Over 1 million larger than 1 km in diameter"
            ])
        ], className="mb-4"),
        
        html.Div([
            html.H4("Kuiper Belt", className="text-primary"),
            html.P([
                "The Kuiper belt is a circumstellar disc in the outer Solar System extending from the orbit of Neptune at 30 AU to approximately 50 AU from the Sun. It is similar to the asteroid belt, but is far larger—20 times as wide and 20–200 times as massive."
            ]),
            html.P([
                "• Distance from Sun: 30 to 50 AU",
                html.Br(),
                "• Notable objects: Pluto, Haumea, Makemake"
            ])
        ], className="mb-4"),
        
        html.Div([
            html.H4("Heliopause", className="text-primary"),
            html.P([
                "The heliopause is the theoretical boundary where the Sun's solar wind is stopped by the interstellar medium; where the solar wind's strength is no longer great enough to push back the stellar winds of the surrounding stars."
            ]),
            html.P([
                "• Distance from Sun: Approximately 120 AU",
                html.Br(),
                "• First crossed by: Voyager 1 in 2012"
            ])
        ], className="mb-4"),
        
        html.Div([
            html.H4("Oort Cloud", className="text-primary"),
            html.P([
                "The Oort cloud, sometimes called the Öpik–Oort cloud, is a theoretical cloud of predominantly icy planetesimals proposed to surround the Sun at distances ranging from 2,000 to 100,000 AU."
            ]),
            html.P([
                "• Distance from Sun: 2,000 to 100,000 AU",
                html.Br(),
                "• Source of: Many long-period comets"
            ])
        ], className="mb-4")
    ], className="p-4 border rounded"),
    
    html.Div([
        html.H3("Astronomical Units (AU)", className="mt-5 mb-3"),
        html.P([
            "An Astronomical Unit (AU) is a unit of length, roughly the distance from Earth to the Sun. One AU is approximately 150 million kilometers (93 million miles)."
        ]),
        html.P([
            "To put the solar system in perspective:",
            html.Br(),
            "• Mercury: 0.39 AU",
            html.Br(),
            "• Venus: 0.72 AU", 
            html.Br(),
            "• Earth: 1 AU",
            html.Br(),
            "• Mars: 1.5 AU",
            html.Br(),
            "• Jupiter: 5.2 AU", 
            html.Br(),
            "• Saturn: 9.5 AU",
            html.Br(),
            "• Uranus: 19.8 AU", 
            html.Br(),
            "• Neptune: 30.1 AU",
            html.Br(),
            "• Pluto (average): 39.5 AU"
        ])
    ], className="p-4 border rounded mt-4")
])