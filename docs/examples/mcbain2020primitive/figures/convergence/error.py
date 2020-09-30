from pathlib import Path
from sys import argv

from matplotlib.pyplot import subplots
import numpy as np
import pandas

from convergence import Convergence

c = {
    csv.stem: pandas.read_csv(csv, index_col=0)
    for csv in (Path(__file__).with_name(a) for a in argv[1:])
}

fig, ax = subplots(figsize=(5, 4))

c_kirchner = {
    "Mamou & Khalid": 0.23752648882047 + 0.003739670622979582j,
    "Paredes et al": 0.2375264888204682 + 0.0037396706229799j,
}["Paredes et al"]

for (k, v), marker in zip(c.items(), "ov"):
    v["c"] = v.creal + 1j * v.cimag

    richardson = Convergence()
    richardson.add_grids(np.vstack([v.index, v.creal]).T)
    with open(Path(__file__).with_name(f"{k}-real-verify.txt"), "w") as fout:
        print(richardson, file=fout)

    p = 4.0
    h = np.array(v.c.index)
    step_ratio = h[:-1] / h[1:]
    c0 = np.array(v.c)
    v["extrapolation"] = pandas.Series(
        c0[1:] + np.diff(c0) / (step_ratio ** p - 1), index=h[1:]
    )  # de Vahl Davis (1986, §4.16.1)

    v.to_csv(Path(__file__).with_name(f"{k}-richardson.csv"))

    error = abs(v.c - c_kirchner)
    ax.loglog(v.index, error, marker=marker, label=k)
    ax.loglog(
        h[1:-2],
        abs(v.extrapolation.iloc[1:-2] - c_kirchner),
        color=ax.lines[-1].get_color(),
        marker=ax.lines[-1].get_marker(),
        linestyle="--",
        markerfacecolor="w",
        label=f"{k} (extrapolated)",
    )

ax.set_xlabel(r"element size, $h$")
ax.set_ylabel(r"error, $|c(h) - c(0)|$")
ax.legend(loc=2)
fig.savefig(Path(__file__).with_suffix(".png"), bbox_inches="tight")
