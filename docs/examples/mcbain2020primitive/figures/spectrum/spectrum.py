from pathlib import Path
from sys import argv

from mpl_toolkits.axisartist.axislines import SubplotZero
from matplotlib.pyplot import figure
import numpy as np
from scipy.constants import golden

c = {Path(c).stem: np.loadtxt(c, dtype=complex) for c in argv[1:]}

fig = figure(figsize=(5, 5 / golden))
ax = SubplotZero(fig, 111)
fig.add_subplot(ax)

for (k, v), marker in zip(c.items(), "xov"):
    (l,) = ax.plot(
        v.real, v.imag, label=k, linestyle="None", marker=marker, markerfacecolor="w"
    )
    if k == "Criminale":
        l.set_markersize(1.2 * l.get_markersize())

ax.set_xlabel(r"real wavespeed, $\Re c$")
ax.set_xlim((0, 1))
ax.set_ylabel(r"imaginary wavespeed, $\Im c$")
ax.set_ylim((-0.8, 0.1))
ax.grid(True)
ax.legend(loc=3)
fig.savefig(Path(__file__).with_suffix(".png"), bbox_inches="tight")
