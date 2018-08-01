"""
Modification of ex09.py to use meshzoo.cube.
"""

from skfem import *
from skfem.models.poisson import *

from meshzoo import cube
import numpy as np

p = np.linspace(0, 1, 16)
# m = MeshHex.init_tensor(p, p, p)
m = MeshTet(*map(np.transpose, cube(nx=len(p), ny=len(p), nz=len(p))))

# e = ElementHex1()
e = ElementTetP1()

map = MappingIsoparametric(m, e)
basis = InteriorBasis(m, e, map)

A = asm(laplace, basis)
b = asm(unit_load, basis)

I = m.interior_nodes()

x = 0*b

if __name__ == "__main__":
    verbose = True
else:
    verbose = False
# run conjugate gradient with the default preconditioner
x[I] = solve(*condense(A, b, I=I), solver=solver_iter_pcg(verbose=verbose))

# run conjugate gradient with the incomplete LU preconditioner
Aint, bint = condense(A, b, I=I)
x[I] = solve(Aint, bint, solver=solver_iter_pcg(pc=build_pc_ilu(Aint), verbose=verbose))

if verbose:
    m.save("ex9_meshzoo.vtk", x)
