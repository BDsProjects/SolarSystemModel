import plotly.graph_objects as go  

import numpy as np  

# Parameters for the ellipse 

a = 2 # semi-major axis 

b = 1 # semi-minor axis 

h = 0 # center x-coordinate 

k = 0 # center y-coordinate  

angle = 0 # rotation angle in radians 

 

# Generate points on the ellipse 

t = np.linspace(0, 2*np.pi, 100)  

x = h + a*np.cos(t)*np.cos(angle) - b*np.sin(t)*np.sin(angle)  

y = k + a*np.cos(t)*np.sin(angle) + b*np.sin(t)*np.cos(angle)  

# Create figure  

fig = go.Figure()  

# Add ellipse as a scatter plot  

fig.add_trace( go.Scatter( x=x, y=y, mode="lines", line=dict(color="blue", width=2), fill="toself", fillcolor="lightblue", opacity=0.5, name="Ellipse" ) ) 

 

# Set axes properties and title  

fig.update_layout( title="Parametric Ellipse", xaxis_title="X", yaxis_title="Y", xaxis=dict(range=[-3, 3]), yaxis=dict(range=[-3, 3]), showlegend=True ) 

fig.show() 