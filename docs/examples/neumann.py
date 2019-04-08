from skfem import *
from skfem.models.poisson import laplace, mass, unit_load

import numpy as np

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

temperature = {s: np.zeros(basis.N) for s in mesh.subdomains}

@linear_form
def interflux(v, dv, w):
    return v * w[0]


# temperature['wire'][closed['wire']] = solve(*condense(
#     thermal_conductivity['wire'] * L + H,
#     f + asm(interflux, interface_basis, w=interface_basis.interpolate(embed(compatible_flux(flux)))),
#     I=closed['wire']))


def mismatch(t: Dict[str, np.ndarray]) -> np.ndarray:
    return t['wire'][interface_dofs] - t['insulation'][interface_dofs]


def poincare_steklov(flux: np.ndarray) -> np.ndarray:
    """return the mismatch in temperature corresponding to a given heat flux"""
    # TODO: Parallelize
    temperature['wire'][closed['wire']] = solve(*condense(
        thermal_conductivity['wire'] * L + H,
        f + asm(interflux, interface_basis,
                w=interface_basis.interpolate(embed(compatible_flux(flux)))),
        I=closed['wire']))
    temperature['insulation'][closed['insulation']] = solve(*condense(
        thermal_conductivity['insulation'] * L,
        -asm(interflux, interface_basis,
             w=interface_basis.interpolate(embed(compatible_flux(flux)))),
        I=closed['insulation']))
    # END TODO
    return mismatch(temperature)


flux = np.arange(len(interface_dofs)) / 2
print(flux)
print(poincare_steklov(flux))
    

if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    mesh.plot(temperature['wire'], colorbar=True)
    mesh.savefig(splitext(argv[0])[0] + '_solution.png')
