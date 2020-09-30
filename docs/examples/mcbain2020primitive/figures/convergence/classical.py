from csv import DictWriter
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


alpha = 1.0
reynolds = 1e4
jare = 1j * alpha * reynolds

with open(Path(__file__).with_suffix(".csv"), "w") as csvfile:

    writer = DictWriter(csvfile, ["h", "creal", "cimag", "condition"])
    writer.writeheader()

    for k in range(5, 12):
        m = skfem.MeshLine(np.linspace(0, 1, 2 ** k))
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

        lap = skfem.asm(laplace, basis)
        M = skfem.asm(mass, basis)

        velocity = Polynomial([1, 0, -1])
        U = basis.interpolate(skfem.project(velocity, basis, basis))
        Upp = basis.interpolate(skfem.project(velocity.deriv(2), basis, basis))

        pencil = skfem.condense(
            (skfem.asm(bilinf, basis) + 2 * alpha ** 2 * lap + alpha ** 4 * M) / jare
            + (alpha ** 2) * skfem.asm(velocity_term, basis, velocity=U)
            + skfem.asm(shear_term, basis, velocity=U)
            + skfem.asm(curvature_term, basis, curvature=Upp,),
            lap + alpha ** 2 * M,
            D=D,
            expand=False,
        )
        c, R = eigs(pencil[0], M=pencil[1], k=2 ** 5, sigma=0.0)

        cargmax = c.imag.argmax()

        cmax = c[cargmax]

        cH, L = eigs(pencil[0].H, M=pencil[1], k=2 ** 5, sigma=0.0)
        right = R[:, cargmax]
        left = L[:, cH.imag.argmin()]
        condition = (
            np.linalg.norm(right)
            * np.linalg.norm(left)
            / np.sqrt(sum(abs(np.vdot(left, p @ right)) ** 2 for p in pencil))
        )

        writer.writerow(
            {
                "h": m.param(),
                "creal": cmax.real,
                "cimag": cmax.imag,
                "condition": condition,
            }
        )
