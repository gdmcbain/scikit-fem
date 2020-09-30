from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.sparse.linalg import eigs

import skfem
from skfem.helpers import dd, dot, ddot
from skfem.models.poisson import laplace, mass


@skfem.BilinearForm
def bilinf(u, v, w):
    return ddot(dd(u), dd(v))


@skfem.BilinearForm
def velocity_term(u, v, w):
    return u * w["velocity"] * v


@skfem.BilinearForm
def shear_term(u, v, w):
    from skfem.helpers import d

    return dot(d(u), w["velocity"] * d(v) + d(w["velocity"]) * v)


@skfem.BilinearForm
def curvature_term(u, v, w):
    return v * w["curvature"] * u


m = skfem.MeshLine(np.linspace(0, 1, 2 ** 6))
e = skfem.ElementLineHermite()
basis = skfem.InteriorBasis(m, e)

D = np.concatenate(
    [
        basis.find_dofs({"centre": m.facets_satisfying(lambda x: x[0] == 0.0)})[
            "centre"
        ].nodal["u_x"],
        basis.find_dofs({"wall": m.facets_satisfying(lambda x: x[0] == 1.0)})[
            "wall"
        ].all(),
    ]
)

alpha = 1.0
reynolds = 1e4
jare = 1j * alpha * reynolds

lap = skfem.asm(laplace, basis)
M = skfem.asm(mass, basis)

velocity = Polynomial([1, 0, -1])
U = basis.interpolate(skfem.project(velocity, basis, basis))
Upp = basis.interpolate(skfem.project(velocity.deriv(2), basis, basis))

A, B = skfem.condense(
    (skfem.asm(bilinf, basis) + 2 * alpha ** 2 * lap + alpha ** 4 * M) / jare
    + (alpha ** 2) * skfem.asm(velocity_term, basis, velocity=U)
    + skfem.asm(shear_term, basis, velocity=U)
    + skfem.asm(curvature_term, basis, curvature=Upp,),
    lap + alpha ** 2 * M,
    D=D,
    expand=False,
)


c = eigs(A, k=2 ** 5, M=B, sigma=0.0, return_eigenvectors=False)

np.savetxt(Path(__file__).with_suffix(".csv"), c)
