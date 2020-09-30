from pathlib import Path

from mpl_toolkits.axisartist.axislines import SubplotZero
from matplotlib.pyplot import figure
import numpy as np

"""
https://matplotlib.org/gallery/axisartist/demo_axisline_style.html

"""


def velocity(z: float) -> float:
    return 1.0 - z ** 2


zrange = (-1.0, 1.0)
z = np.linspace(*zrange)

fig = figure(figsize=(5, 2.5))
ax = SubplotZero(fig, 111)
fig.add_subplot(ax)

for direction, axis in ax.axis.items():
    axis.set_axisline_style("-|>")
    axis.set_visible(direction.endswith("zero"))

ax.plot(z, velocity(z))

for z in zrange:
    ax.axvline(x=z, linestyle="--")

ax.axis["xzero"].set_label(r"$z$")
ax.axis["yzero"].set_label(r"$x$")
z_annotate = 0.3
ax.annotate(r"$U(z) = 1 - z^2$", (z_annotate, velocity(z_annotate)))
ax.set_xlim(zrange)
ax.set_ylim(0.0)
ax.set_xticks(np.linspace(*zrange, 5))
ax.set_yticks([])
fig.savefig(Path(__file__).with_suffix(".png"), bbox_inches="tight")
