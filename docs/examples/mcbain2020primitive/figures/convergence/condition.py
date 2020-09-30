from pathlib import Path
from sys import argv

from mpl_toolkits.axisartist.axislines import SubplotZero
from matplotlib.pyplot import figure
import pandas
from scipy.constants import golden

c = {
    csv.stem: pandas.read_csv(csv, index_col=0)
    for csv in (Path(__file__).with_name(a) for a in argv[1:])
}

fig = figure(figsize=(5, 5 / golden))
ax = SubplotZero(fig, 111)
fig.add_subplot(ax)

for (k, v), marker in zip(c.items(), "ov"):
    ax.loglog(v.index, v.condition, marker=marker, label=k)

ax.set_xlabel(r"element size, $h$")
ax.set_ylabel("condition")
ax.legend(loc=2)
fig.savefig(Path(__file__).with_suffix(".png"), bbox_inches="tight")
