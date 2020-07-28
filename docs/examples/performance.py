"""A simple performance test.

This script is used to generate the table in README.md.

"""
from timeit import timeit
import numpy as np
from skfem import *
from skfem.models.poisson import laplace, unit_load


def pre(N=3):
    m = MeshTet.init_tensor(*(3 * (np.linspace(0., 1., N),)))
    return m


print('| Degrees-of-freedom | Time spent in assembly (s) | Time spent in linear solve (s) |')
print('| --- | --- | --- |')


for k in range(4, 20):

    m = pre(int(2**(k/3)))

    def assembler(m):
        basis = InteriorBasis(m, ElementTetP1())
        return (asm(laplace, basis),
                asm(unit_load, basis),
                m.boundary_nodes())

    A, b, D = assembler(m)

    assemble_time = timeit(lambda: assembler(m), number=1) / 1.

    def solver(A, b):
        return solve(*condense(A, b, D=D))

    if A.shape[0] < 5e4:
        solve_time = timeit(lambda: solver(A, b), number=1) / 1.
    else:
        solve_time = np.nan
        
    print('| {} | {} | {} |'.format(len(b), assemble_time, solve_time))
