from skfem import *
from skfem.models.poisson import laplace, mass, unit_load

import numpy as np
from scipy.optimize import root
from scipy.sparse.linalg import LinearOperator, cg

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

from typing import Optional, Dict


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
wire_basis = InteriorBasis(mesh, element, elements=mesh.subdomains['wire'])

@functional
def unit_integrand(w):
    return 1.


discrete_net_generation = joule_heating * sum(asm(unit_integrand, wire_basis))
print('Net generation: {0} (cf. exact {1})'.format(discrete_net_generation,
                                                   joule_heating * np.pi * radii[0]**2))



closed = {s: np.unique(basis.element_dofs[:, d])
          for s, d in mesh.subdomains.items()}
interface_dofs = np.intersect1d(*closed.values())
interface_facets = np.logical_and(*np.isin(mesh.facets, interface_dofs))
interface_basis = FacetBasis(mesh, element,
                             facets=np.nonzero(interface_facets)[0])
interior = {s: np.setdiff1d(d, interface_dofs) for s, d in closed.items()}

outside_basis = FacetBasis(mesh, element,
                           facets=mesh.boundaries['convection'])


@functional
def net_interflux(w):
    return w[0]


def embed(v: np.ndarray) -> np.ndarray:
    w = np.zeros(basis.N)
    w[interface_dofs] = v
    return w


flux = np.ones(len(interface_dofs))
unit_flux = sum(asm(net_interflux, interface_basis, w=interface_basis.interpolate(embed(flux))))
print('Net flux: {0} (cf. exact {1})'.format(
    unit_flux,
    2 * np.pi * radii[0]))


def incompatibility(flux: np.ndarray) -> float:
    return (sum(asm(net_interflux, interface_basis, w=interface_basis.interpolate(embed(flux))))
            - discrete_net_generation)


def compatible_flux(flux: np.ndarray) -> np.ndarray:
    return flux - incompatibility(flux) * np.ones(len(interface_dofs)) / unit_flux
    

print('Incompatibility: {}'.format(
    incompatibility(flux)))
print('Incompatibility: {}'.format(
    incompatibility(compatible_flux(flux))))

L = asm(laplace, basis)
f = joule_heating * asm(unit_load, basis)
H = heat_transfer_coefficient * asm(mass, outside_basis)


@linear_form
def interflux(v, dv, w):
    return v * w[0]



def mismatch(t: Dict[str, np.ndarray]) -> np.ndarray:
    return t['wire'][interface_dofs] - t['insulation'][interface_dofs]



def solve_temperatures(flux: np.ndarray,
                       eps: float=1e-3) -> Dict[str, np.ndarray]:
    temperature = {'insulation': np.zeros(basis.N)}
    flux1 = embed(compatible_flux(flux))
    ff = asm(interflux, interface_basis, w=interface_basis.interpolate(flux1))
    temperature['insulation'][closed['insulation']] = solve(*condense(
        thermal_conductivity['insulation'] * L + H, -ff, I=closed['insulation']))
    temperature['wire'] = np.zeros_like(temperature['insulation'])
    temperature['wire'][closed['wire']] = solve(*condense(thermal_conductivity['wire'] * L,
                                                          f + ff,
                                                          I=closed['wire']))
    temperature['wire'] -= (temperature['wire'][interface_dofs[0]]
                            - temperature['insulation'][interface_dofs[0]])
    return temperature

    
def poincare_steklov(flux: np.ndarray) -> np.ndarray:
    """return the mismatch in temperature corresponding to a given heat flux"""
    return mismatch(solve_temperatures(flux))


A0 = poincare_steklov(np.zeros(len(interface_dofs)))
print('A0:', A0)
A = LinearOperator((len(interface_dofs),) * 2, lambda q: poincare_steklov(q) - A0, dtype=float)
solution = cg(A, np.zeros(len(interface_dofs)))                   
print('solution:', solution)
flux = solution.x
print('flux:', flux)
temperature = solve_temperatures(flux)
print('wire temperature:', temperature['wire'][closed['wire']])
print('insulation temperature:', temperature['insulation'][closed['insulation']])


if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    T0 = {'skfem': basis.interpolator(temperature['wire'])(np.zeros((2, 1)))[0],
          'exact':
          (joule_heating * radii[0]**2 / 4 / thermal_conductivity['wire'] *
           (2 * thermal_conductivity['wire'] / radii[1]
            / heat_transfer_coefficient
            + (2 * thermal_conductivity['wire']
               / thermal_conductivity['insulation']
               * np.log(radii[1] / radii[0])) + 1))}

    print('Central temperature:', T0)

    for s in mesh.subdomains:
        mesh.plot(temperature[s], colorbar=True)
        mesh.savefig(splitext(argv[0])[0] + f'_{s}_solution.png')
