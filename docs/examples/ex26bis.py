r"""Restricting a problem to a subdomain.

The `ex17.py` example solved the steady-state heat equation with uniform
volumetric heating in a central core surrounded by an annular insulating layer
of lower thermal conductivity.  Here, the problem is completely restricted to
the core, taking the temperature as zero throughout the annulus.

Thus the problem reduces to the same Poisson equation with uniform forcing and
homogeneous Dirichlet conditions:

.. math::
   \nabla\cdot(k\nabla T) + A  = 0, \qquad 0 < r < a
with

.. math::
   T = 0, \qquad\text{on}\quad r = a.
The exact solution is

.. math::
   T = \frac{s}{4k}(a^2 - r^2).

The novelty here is that the temperature is defined as a finite element function
throughout the mesh (:math:`r < b`) but only solved on a subdomain.

"""
from re import U
from skfem import *
from skfem.models.poisson import laplace, unit_load

from typing import Tuple

import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import LinearOperator, minres

from ex17 import mesh, basis, radii, joule_heating, thermal_conductivity, temperature, H


@Functional
def heat(w):
    from skfem.helpers import dot

    return dot(w.n, w.u.grad)


dofs = {s: np.unique(basis.element_dofs[:, e]) for s, e in mesh.subdomains.items()}

bases = {
    s: CellBasis(basis.mesh, basis.elem, elements=e) for s, e in mesh.subdomains.items()
}

L = {s: laplace.assemble(b) for s, b in bases.items()}
f = asm(unit_load, bases["core"])

interfacial_dofs = np.intersect1d(*dofs.values())
interface = [
    FacetBasis(mesh, basis.elem, facets=mesh.facets_around("core", flip=flip))
    for flip in range(2)
]


def subdomains(interfacial_temperature: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = basis.zeros()
    x[interfacial_dofs] = interfacial_temperature
    return (
        solve(
            *condense(
                thermal_conductivity["core"] * L["core"],
                joule_heating * f,
                x=x,
                D=dofs["annulus"],
            )
        ),
        solve(
            *condense(
                thermal_conductivity["annulus"] * L["annulus"] + H, x=x, D=dofs["core"]
            )
        ),
    )


def steklov(interfacial_temperature: np.ndarray) -> np.ndarray:
    temperatures = subdomains(interfacial_temperature)
    heat_flux = np.array(
        [
            -k * heat.elemental(b, u=u)
            for k, b, u in zip(thermal_conductivity.values(), interface, temperatures)
        ]
    )
    return heat_flux.sum(0)


steklov0 = steklov(np.zeros(interfacial_dofs.size))
K = LinearOperator(
    (steklov0.size,) * 2, lambda t: steklov(t) - steklov0, dtype=steklov0.dtype
)

tint, exit_code = minres(K, -steklov0, np.zeros_like(temperature[interfacial_dofs]))
print(f"{exit_code=}")

# temperatures = subdomains(temperature[interfacial_dofs])
print(temperature[interfacial_dofs])
print(tint)
temperatures = subdomains(tint)

T0 = {
    "skfem": (basis.probes(np.zeros((2, 1))) @ temperatures[0])[0],
    "exact": joule_heating * radii[0] ** 2 / 4 / thermal_conductivity["core"],
}

if __name__ == "__main__":
    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import draw, plot

    ax = draw(mesh)
    plot(
        mesh,
        np.array(temperatures).max(0)[basis.nodal_dofs.flatten()],
        ax=ax,
        colorbar=True,
    )
    ax.get_figure().savefig(splitext(argv[0])[0] + "_solution.png")

    ax = draw(mesh)
    plot(mesh, temperatures[0][basis.nodal_dofs.flatten()], ax=ax, colorbar=True)
    ax.get_figure().savefig(splitext(argv[0])[0] + "_core_solution.png")

    ax = draw(mesh)
    plot(mesh, temperatures[1][basis.nodal_dofs.flatten()], ax=ax, colorbar=True)
    ax.get_figure().savefig(splitext(argv[0])[0] + "_annular_solution.png")
