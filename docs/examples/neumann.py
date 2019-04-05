from typing import Optional

import numpy as np

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

from skfem import *
from skfem.models.poisson import laplace, mass, unit_load

radii = [2., 3.]
joule_heating = 5.
heat_transfer_coefficient = 7.
thermal_conductivity = {'wire': 101.,  'insulation': 11.}


def make_mesh(a: float,         # radius of wire
              b: float,         # radius of insulation
              dx: Optional[float] = None) -> MeshTri:

    dx = a / 2 ** 3 if dx is None else dx

    origin = np.zeros(3)
    geom = Geometry()
    wire = geom.add_circle(origin, a, dx, make_surface=True)
    geom.add_physical(wire.plane_surface, 'wire')
    insulation = geom.add_circle(origin, b, dx, holes=[wire.line_loop])
    geom.add_physical(insulation.plane_surface, 'insulation')
    geom.add_physical(insulation.line_loop.lines, 'convection')

    return MeshTri.from_meshio(generate_mesh(geom, dim=2))


mesh = make_mesh(*radii)

element = ElementTriP1()
basis = InteriorBasis(mesh, element)

closed = {s: np.unique(basis.element_dofs[:, d])
          for s, d in mesh.subdomains.items()}
interface_dofs = np.intersect1d(*closed.values())
interface_facets = np.logical_and(*np.isin(mesh.facets, interface_dofs))
interface_basis = FacetBasis(mesh, element,
                             facets=np.nonzero(interface_facets)[0])
interior = {s: np.setdiff1d(d, interface_dofs) for s, d in closed.items()}

outside_basis = FacetBasis(mesh, element,
                           facets=mesh.boundaries['convection'])

L = asm(laplace, basis)
f = joule_heating * asm(unit_load, basis)
H = heat_transfer_coefficient * asm(mass, outside_basis)

temperature = np.zeros(basis.N)

## Cheat for the moment.  Assuming radial symmetry, the flux across
## the interface into the insulation has to be uniform and can be
## deduced from the integral of the Joule heating over the wire as
## radii[0] * joule_heating / 2.

temperature[closed['insulation']] = solve(*condense(
    thermal_conductivity['insulation'] * L + H,
    radii[0] * joule_heating * asm(unit_load, interface_basis),
    I=closed['insulation']))
    

if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    mesh.plot(temperature, colorbar=True)
    mesh.savefig(splitext(argv[0])[0] + '_solution.png')
