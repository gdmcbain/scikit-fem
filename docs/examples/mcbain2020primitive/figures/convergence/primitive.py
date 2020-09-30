from csv import DictWriter
from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.sparse import block_diag, bmat, csr_matrix
from scipy.sparse.linalg import eigs

import skfem
from skfem.models.general import divergence
from skfem.models.poisson import laplace, mass


U = Polynomial([1, 0, -1])  # base-flow profile


@skfem.BilinearForm
def base_velocity(u, v, w):
    return v * U(w.x[0]) * u


@skfem.BilinearForm
def base_shear(u, v, w):
    return v * U.deriv()(w.x[0]) * u


alpha = 1.0
re = 1e4
jare = 1j * alpha * re

with open(Path(__file__).with_suffix(".csv"), "w") as csvfile:

    writer = DictWriter(csvfile, ["h", "creal", "cimag", "condition"])
    writer.writeheader()

    for k in range(5, 15):
        element = {"u": skfem.ElementLineMini(), "p": skfem.ElementLineP1()}
        basis = {
            v: skfem.InteriorBasis(
                skfem.MeshLine(np.linspace(0, 1, 2 ** k)), e, intorder=4
            )
            for v, e in element.items()
        }

        L = skfem.asm(laplace, basis["u"])
        M = skfem.asm(mass, basis["u"])
        P = skfem.asm(mass, basis["u"], basis["p"])
        B = -skfem.asm(divergence, basis["u"], basis["p"])
        V = skfem.asm(base_velocity, basis["u"])
        W = skfem.asm(base_shear, basis["u"])

        Kuu = jare * V + alpha ** 2 * M + L
        stiffness = bmat(
            [
                [Kuu, re * W, jare * P.T],
                [None, Kuu, re * B.T],
                [-jare * P, re * B, None],
            ],
            "csc",
        )
        mass_matrix = block_diag([M, M, csr_matrix((basis["p"].N,) * 2)], "csr")

        # Seek only 'even' modes, 'even' in terms of conventional
        # stream-function formulation, so that the longitudinal component u of
        # the perturbation to the velocity vanishes on the centre-line z = 0,
        # z here being the sole coordinate.

        u_boundaries = basis["u"].find_dofs()["all"].all()
        walls = np.concatenate([u_boundaries, u_boundaries[1:] + basis["u"].N])

        pencil = skfem.condense(stiffness, mass_matrix, D=walls, expand=False)
        c, R = eigs(pencil[0], M=pencil[1], k=2 ** 5, sigma=0.0)
        c /= jare
        cargmax = c.imag.argmax()
        cmax = c[cargmax]

        cH, L = eigs(pencil[0].H, M=pencil[1], k=len(c), sigma=0.0)
        cH /= -jare
        right = R[:, cargmax]
        left = L[:, cH.imag.argmin()]
        condition = (
            np.linalg.norm(right)
            * np.linalg.norm(left)
            / np.sqrt(sum(abs(np.vdot(left, p @ right)) ** 2 for p in pencil))
        )

        writer.writerow(
            {
                "h": basis["u"].mesh.param(),
                "creal": cmax.real,
                "cimag": cmax.imag,
                "condition": condition,
            }
        )
