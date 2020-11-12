from skfem import *

import numpy as np

import meshio

from ex02 import m, ib, dofs, x

bottom_facets = m.facets_satisfying(lambda x: x[1] == 0)
dofs["bottom"] = ib.find_dofs({"bottom": bottom_facets})["bottom"]
bb = FacetBasis(ib.mesh, ib.elem, facets=bottom_facets)

points = bb.doflocs[0, dofs["bottom"].nodal["u"]]
lines_global = bb.element_dofs[
    np.isin(bb.element_dofs, dofs["bottom"].nodal["u"]).reshape((-1, bb.nelems))
]
lines_local = np.unique(lines_global, return_inverse=True)[1].reshape((-1, bb.nelems))

x1 = x[dofs["bottom"].nodal["u"]]


mesh = MeshLine(points, lines_local)
print(mesh)

l1b = InteriorBasis(mesh, ElementLineP1())
l0b = InteriorBasis(mesh, ElementLineP0())
x0 = project(x1, l1b, l0b)

mio = meshio.Mesh(
    np.outer(points, [1, 0, 0]),
    [("line", lines_local.T)],
    point_data={"deflexion": x1},
    cell_data={"deflexion": [x0]},
)
mio.write("bottom.xdmf")
