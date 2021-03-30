from typing import Callable, Optional, Tuple

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix
from skfem.element import DiscreteField, Element
from skfem.mapping import Mapping
from skfem.mesh import Mesh

from .basis import Basis


class InteriorBasis(Basis):
    """Basis functions evaluated at quadrature points inside the elements.

    :class:`~skfem.assembly.InteriorBasis` object is a combination of
    :class:`~skfem.mesh.Mesh` and :class:`~skfem.element.Element`:

    >>> from skfem import *
    >>> m = MeshTri.init_symmetric()
    >>> e = ElementTriP1()
    >>> basis = InteriorBasis(m, e)

    The resulting objects are used in the assembly.

    >>> from skfem.models.poisson import laplace
    >>> K = asm(laplace, basis)
    >>> K.shape
    (5, 5)

    """
    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: Optional[Mapping] = None,
                 intorder: Optional[int] = None,
                 elements: Optional[ndarray] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None):
        """Combine :class:`~skfem.mesh.Mesh` and :class:`~skfem.element.Element`
        into a set of precomputed global basis functions.

        Parameters
        ----------
        mesh
            An object of type :class:`~skfem.mesh.Mesh`.
        elem
            An object of type :class:`~skfem.element.Element`.
        mapping
            An object of type :class:`skfem.mapping.Mapping`. If `None`, uses
            `mesh.mapping`.
        intorder
            Optional integration order, i.e. the degree of polynomials that are
            integrated exactly by the used quadrature. Not used if `quadrature`
            is specified.
        elements
            Optional subset of element indices.
        quadrature
            Optional tuple of quadrature points and weights.

        """
        super(InteriorBasis, self).__init__(mesh,
                                            elem,
                                            mapping,
                                            intorder,
                                            quadrature,
                                            mesh.refdom)

        self.basis = [self.elem.gbasis(self.mapping, self.X, j, tind=elements)
                      for j in range(self.Nbfun)]

        if elements is None:
            self.nelems = mesh.nelements
        else:
            self.nelems = len(elements)
        self.tind = elements

        self.dx = (np.abs(self.mapping.detDF(self.X, tind=elements))
                   * np.tile(self.W, (self.nelems, 1)))

    def default_parameters(self):
        """Return default parameters for `~skfem.assembly.asm`."""
        return {'x': self.global_coordinates(),
                'h': self.mesh_parameters()}

    def global_coordinates(self) -> DiscreteField:
        return DiscreteField(self.mapping.F(self.X, tind=self.tind))

    def mesh_parameters(self) -> DiscreteField:
        return DiscreteField(np.abs(self.mapping.detDF(self.X, self.tind))
                             ** (1. / self.mesh.dim()))

    def refinterp(self,
                  y: ndarray,
                  nrefs: int = 1,
                  Nrefs: Optional[int] = None) -> Tuple[Mesh, ndarray]:
        """Refine and interpolate (for plotting)."""
        if Nrefs is not None:
            nrefs = Nrefs  # for backwards compatibility
        # mesh reference domain, refine and take the vertices
        meshclass = type(self.mesh)
        m = meshclass.init_refdom().refined(nrefs)
        X = m.p

        # map vertices to global elements
        x = self.mapping.F(X)

        # interpolate some previous discrete function at the vertices
        # of the refined mesh
        w = 0. * x[0]
        for j in range(self.Nbfun):
            basis = self.elem.gbasis(self.mapping, X, j)
            w += y[self.element_dofs[j]][:, None] * basis[0]

        # create connectivity for the new mesh
        nt = self.nelems
        t = np.tile(m.t, (1, nt))
        dt = np.max(t)
        t += (dt + 1) *\
            (np.tile(np.arange(nt), (m.t.shape[0] * m.t.shape[1], 1))
             .flatten('F')
             .reshape((-1, m.t.shape[0])).T)

        if X.shape[0] == 1:
            p = np.array([x.flatten()])
        else:
            p = x[0].flatten()
            for itr in range(len(x) - 1):
                p = np.vstack((p, x[itr + 1].flatten()))

        M = meshclass(p, t, validate=False)

        return M, w.flatten()

    def probes_data(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return a values of basis functions and relevant indices
        for interpolating a solution at the given points `x`."""

        finder = self.mesh.element_finder(mapping=self.mapping)
        cells = finder(*x)
        pts = self.mapping.invF(x[:, :, np.newaxis], tind=cells)
        phis = np.array([
            self.elem.gbasis(self.mapping, pts, k, tind=cells)[0][0].flatten()
            for k in range(self.Nbfun)
        ])
        indices = self.element_dofs[:, cells]
        return phis, indices

    def probes_matrix(self, x: np.ndarray) -> coo_matrix:
        """Return a coo_matrix which acts on a solution to probe its values
        at the given points `x`"""
        phis, indices = self.probes_data(x)
        return coo_matrix(
            (
                phis.flatten(),
                (np.tile(np.arange(x.shape[1]), self.Nbfun), indices.flatten()),
            ),
            shape=(x.shape[1], self.N),
        )

    def probes(self, x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """Return a function handle, which can be used for finding
        the values on points `x` of a given solution vector."""

        phis, indices = self.probes_data(x)

        def interpolator(y: np.ndarray) -> np.ndarray:
            return np.sum(phis * y[indices], 0)

        return interpolator

    def interpolator(self, y: ndarray) -> Callable[[ndarray], ndarray]:
        """Return a function handle, which can be used for finding
        values of the given solution vector `y` on given points."""

        def interpfun(x: np.ndarray) -> np.ndarray:
            return self.probes(x)(y)

        return interpfun

    def with_element(self, elem: Element) -> 'InteriorBasis':
        """Return a similar basis using a different element."""
        return type(self)(
            self.mesh,
            elem,
            mapping=self.mapping,
            quadrature=self.quadrature,
            elements=self.tind,
        )
