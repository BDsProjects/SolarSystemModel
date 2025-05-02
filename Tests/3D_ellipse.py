import plotly.graph_objects as go
import numpy as np

# Parameters for the 3D ellipsoid
a = 2  # semi-axis in x direction
b = 1  # semi-axis in y direction
c = 1.5  # semi-axis in z direction

# Create the mesh grid for the ellipsoid
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = a * np.outer(np.cos(u), np.sin(v))
y = b * np.outer(np.sin(u), np.sin(v))
z = c * np.outer(np.ones_like(u), np.cos(v))

# Create the 3D figure
fig = go.Figure(data=[
    go.Surface(
        x=x, y=y, z=z,
        colorscale='Blues',
        opacity=0.8,
        showscale=False
    )
])

# Update layout for better visualization
fig.update_layout(
    title="3D Ellipsoid in Plotly",
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data',  # this preserves the actual shape of the ellipsoid
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    width=700,
    height=700
)

# Display the figure
fig.show()