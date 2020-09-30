from pathlib import Path

from mpl_toolkits.axisartist.axislines import SubplotZero
from matplotlib.pyplot import figure
import numpy as np

from scipy.constants import golden

import skfem


def hermite(z: float, k: int) -> float:
    return [
        (2 * z + 1) * (z - 1) ** 2,
        z * (z - 1) ** 2,
        (3 - 2 * z) * z ** 2,
        (1 - z) * z ** 2,
    ][k]


zrange = (0.0, 1.0)
z = np.linspace(*zrange)

fig = figure(figsize=(5, 2))
ax = SubplotZero(fig, 111)
fig.add_subplot(ax)

elements = {
    "Hermite": skfem.ElementLineHermite(),
    "Mini": skfem.ElementLineMini(),
    "Taylor–Hood": skfem.ElementLineP2(),
}

for (name, e), linestyle, marker in zip(
    elements.items(), ["-", ":", (0, (9, 16))], [None, "+", "x"]
):
    for k, _ in enumerate(e.doflocs):
        l = ax.plot(
            z,
            hermite(z, k) if name == "Hermite" else e.lbasis(z[:, np.newaxis].T, k)[0],
            color=None if k == 0 else l[0].get_color(),
            linestyle=linestyle,
            marker=None,
            label=name if k == 0 else None,
        )

for direction, axis in ax.axis.items():
    axis.set_axisline_style("-|>")
    axis.set_visible(direction.endswith("zero"))

ax.axis["xzero"].set_label(r"$\zeta$")
ax.axis["yzero"].set_visible(False)

ax.set_xlim(zrange)
ax.set_ylim(0.0)
ax.set_xticks(np.linspace(0, 1, 3))
ax.legend(loc=3)
fig.savefig(Path(__file__).with_suffix(".png"), bbox_inches="tight")
