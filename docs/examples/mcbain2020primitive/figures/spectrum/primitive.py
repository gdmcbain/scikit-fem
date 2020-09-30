from skfem import *
from skfem.models.general import divergence
from skfem.models.poisson import laplace, mass, unit_load

from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.sparse import block_diag, bmat, csr_matrix
from scipy.sparse.linalg import eigs


@BilinearForm
def base_velocity(u, v, w):
    return v * w["U"] * u


@BilinearForm
def base_shear(u, v, w):
    return v * w["U"].grad[0] * u


mesh = MeshLine(np.linspace(0, 1, 2 ** 6))

element = {"u": ElementLineMini(), "p": ElementLineP1()}

basis = {v: InteriorBasis(mesh, e, intorder=4) for v, e in element.items()}

L = asm(laplace, basis["u"])
M = asm(mass, basis["u"])
P = asm(mass, basis["u"], basis["p"])
B = -asm(divergence, basis["u"], basis["p"])

noslip = basis["u"].find_dofs(
    {"noslip": basis["u"].mesh.facets_satisfying(lambda x: x[0] == 1.0)}
)
U = basis["u"].interpolate(
    solve(
        *condense(L, 2 * asm(unit_load, basis["u"]), np.zeros(basis["u"].N), D=noslip)
    )
)

V = asm(base_velocity, basis["u"], U=U)
W = asm(base_shear, basis["u"], U=U)

re = 1e4  # Reynolds number
alpha = 1.0  # longitudinal wavenumber
jare = 1j * alpha * re
Kuu = jare * V + alpha ** 2 * M + L
stiffness = bmat(
    [[Kuu, re * W, jare * P.T], [None, Kuu, re * B.T], [-jare * P, re * B, None]], "csc"
)
mass_matrix = block_diag([M, M, csr_matrix((basis["p"].N,) * 2)], "csr")

# Seek only 'even' modes, 'even' in terms of conventional
# stream-function formulation, so that the longitudinal component u of
# the perturbation to the velocity vanishes on the centre-line z = 0,
# z here being the sole coordinate.

u_boundaries = basis["u"].find_dofs()["all"].all()
walls = np.concatenate([u_boundaries, u_boundaries[1:] + basis["u"].N])

pencil = condense(stiffness, mass_matrix, D=walls, expand=False)
c = eigs(pencil[0], M=pencil[1], k=2 ** 5, sigma=0.0, return_eigenvectors=False) / jare

np.savetxt(Path(__file__).with_suffix(".csv"), c)
