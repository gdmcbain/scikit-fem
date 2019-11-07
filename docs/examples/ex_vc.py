from os.path import splitext
from sys import argv

import numpy as np

from skfem import *

name = splitext(argv[0])[0]

mesh = MeshTri.init_symmetric()
mesh.refine(3)

element = {'u': ElementTriP2(), 'p': ElementTriP1()}
basis = {v: InteriorBasis(mesh, e, intorder=3) for v, e in element.items()}

p = basis['p'].doflocs[0]       # p (x, y) = x
ax = mesh.plot(p)
ax.get_figure().savefig(f'{name}-pressure.png')

u, v = (derivative(p, basis['p'], basis['u'], i) for i in range(2))

ax = mesh.draw()
ax.quiver(*mesh.p, u[basis['u'].nodal_dofs], v[basis['u'].nodal_dofs])
ax.get_figure().savefig(f'{name}-velocity.png')



