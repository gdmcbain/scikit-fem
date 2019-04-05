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

conductivity = basis.zero_w()
for subdomain, elements in mesh.subdomains.items():
    conductivity[elements] = thermal_conductivity[subdomain]

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

# TODO: Solving a Dirichlet problem in the wire and a Neumann problem in the insulation is counterintuitive given the relative thermal conductivities; however, the wire is landlocked so this avoids having to deal with a pure Neumann problem for the moment.

def solve_wire():
    """solve Dirichlet problem in wire"""
    temperature[interior['wire']] = solve(*condense(
        thermal_conductivity['wire'] * L,
        f,
        temperature,
        D=closed['insulation']))


def solve_insulation():
    """solve Neumann problem in insulation"""
    interfacial
    temperature[closed['insulation']] = solve(*condense(
        thermal_conductivity['insulation'] * L + H,
        ...))

solve_wire()


if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    T0 = {'skfem': basis.interpolator(temperature)(np.zeros((2, 1)))[0],
          'exact':
          (joule_heating * radii[0]**2 / 4 / thermal_conductivity['wire'] *
           (2 * thermal_conductivity['wire'] / radii[1]
            / heat_transfer_coefficient
            + (2 * thermal_conductivity['wire']
               / thermal_conductivity['insulation']
               * np.log(radii[1] / radii[0])) + 1))}
    print('Central temperature:', T0)

    ax = mesh.plot(temperature)
    fig = ax.get_figure()
    fig.colorbar(ax.get_children()[0])
    fig.savefig(splitext(argv[0])[0] + '_solution.png')
