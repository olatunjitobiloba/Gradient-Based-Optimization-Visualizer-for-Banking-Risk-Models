import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from loss_landscape import compute_gradient, compute_loss
from optimizer import gradient_descent

st.title("Loss Landscape Explorer")

learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
steps = st.sidebar.slider("Steps", 10, 200, 100)
start_x = st.sidebar.slider("Start x", -5.0, 5.0, -4.0, 0.1)
start_y = st.sidebar.slider("Start y", -5.0, 5.0, 3.0, 0.1)

params, history = gradient_descent(
    compute_loss, compute_gradient, [start_x, start_y], lr=learning_rate, steps=steps
)
history = np.array(history)

grid = np.linspace(-5.12, 5.12, 60)
X, Y = np.meshgrid(grid, grid)
Z = np.array([[compute_loss(np.array([x, y])) for x in grid] for y in grid])

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
ax.plot(history[:, 0], history[:, 1], [compute_loss(p) for p in history], color="red", marker="o")
ax.set_xlabel("weight 1")
ax.set_ylabel("weight 2")
ax.set_zlabel("loss")
st.pyplot(fig)
