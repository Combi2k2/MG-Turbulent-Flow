import torch
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

data = torch.load('rbc_data/sample_0.pt')
data = data.numpy()

fig, ax = plt.subplots()

def animate(i):
    ax.imshow(data[i][0])

ani = FuncAnimation(fig, animate, frames = 1000, interval = 50, repeat = False)
plt.show()