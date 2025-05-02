import plotly.graph_objects as go 

import numpy as np 

  

# Create a figure 

fig = go.Figure() 

  

# Add an ellipse shape 

fig.add_shape( 

    type="circle", 

    xref="x", yref="y", 

    x0=-1, y0=-1, 

    x1=1, y1=1, 

    line_color="blue", 

    fillcolor="lightblue", 

    opacity=0.5 

) 

  

# Set axes properties 

fig.update_xaxes(range=[-2, 2]) 

fig.update_yaxes(range=[-2, 2]) 

  

# Add title and display the figure 

fig.update_layout(title="Ellipse using Plotly") 

fig.show() 