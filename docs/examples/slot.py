from pathlib import Path

from matplotlib.pyplot import subplots
import numpy as np
from scipy.sparse import dia_matrix

import pacopy
from skfem import *
from skfem.models.poisson import unit_load


@linear_form
def young_laplace(v, dv, w):
    return sum(dv * w.dw) / np.sqrt(1 + sum(w.dw**2))

@bilinear_form
def jacobian(u, du, v, dv, w):
    return sum(dv * du) / (1 + sum(w.dw**2))**(3/2)

class Slot:

    def __init__(self, n: int, b: float = 1.):

        self.b = b
        x = np.sin(np.pi * np.linspace(-1., 1., 2 * n + 1) / 2) * b / 2
        self.basis = InteriorBasis(MeshLine(x), ElementLineP1())
        self.I = self.basis.mesh.interior_nodes()
        self.D = self.basis.complement_dofs(self.I)
        self.unit_load = asm(unit_load, self.basis)

    def inner(self, a: np.ndarray, b: np.ndarray) -> float:
        """return the inner product of two solutions"""
        return a.T @ b

    def norm2_r(self, a: np.ndarray) -> float:
        """return the squared norm in the range space

        used to determine if a solution has been found.
        """
        return a.T @ a

    def f(self, u: np.ndarray, kappa: float) -> np.ndarray:
        """return the residual at u"""
        out = (asm(young_laplace, self.basis, w=self.basis.interpolate(u))
               - kappa * self.unit_load)
        out[self.D] = u[self.D]
        return out

    def df_dlmbda(self, u: np.ndarray, kappa: float) -> np.ndarray:
        out = -self.unit_load
        out[self.D] = 0.0
        return out

    def jacobian_solver(self,
                        u: np.ndarray,
                        kappa: float,
                        rhs: np.ndarray) -> np.ndarray:
        """A solver for the Jacobian problem."""
        du = np.zeros_like(u)
        du[self.I] = solve(*condense(
            asm(jacobian, self.basis, w=self.basis.interpolate(u)),
            rhs, I=self.I))
        return du


problem = Slot(2**3)

kappa = []
height = []

upper = 0.5 - 0.01


class RangeException(Exception):
    pass


def callback(k, kapp, sol):

    kappa.append(problem.b * kapp)
    height.append(max(sol) / problem.b)

    if kappa[-1] > 2 or height[-1] > upper:
        raise RangeException
    


try:
    pacopy.natural(problem, np.zeros(problem.basis.N), 0., callback)
except RangeException:
    pass

fig, ax = subplots()
ax.set_title('plane meniscus pinned on a slot')
ax.set_xlabel('reduced mean curvature, $bK_\mathrm{m}$')
ax.set_ylabel('reduced height, $h/b$')
ax.grid()
ax.plot(kappa, height, 'ro', label='skfem')
bKm = np.linspace(0., 2., 2**9)[1:]
ax.plot(bKm, (1. - np.sqrt(1. - (bKm/2)**2)) / bKm, 'g--', label='exact')
ax.set_xlim(0.0, 2.0)
ax.set_ylim(0.0, 0.5)
ax.legend(loc=2)
fig.savefig(Path(__file__).with_suffix('.png'))
