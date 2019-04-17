from ns import *

from scipy.sparse.linalg import splu, spilu, gmres, LinearOperator


def creeping(step):
    """return the solution for zero Reynolds number"""
    uvp = step.make_vector()
    uvp[step.I] = solve(step.lu0,
                        condense(step.S, np.zeros_like(uvp), uvp, step.I)[1],
                        lambda A, b: A.solve(b))
    return uvp


def jacobian_solver(step,
                    uvp: np.ndarray,
                    reynolds: float,
                    rhs: np.ndarray) -> np.ndarray:
    duvp = step.make_vector() - uvp
    u = step.basis['u'].interpolate(step.split(uvp)[0])
    A = (step.S
         + reynolds * block_diag([asm(acceleration_jacobian,
                                      step.basis['u'], w=u),
                                  csr_matrix((step.basis['p'].N,)*2)]))
    A1 = condense(A, I=step.I)
    _, rhs1 = condense(A, rhs, duvp, I=step.I)
    duvp[step.I], info = gmres(A1, rhs1, step.lu0.solve(rhs1), 1e-12,
                               M=build_pc_ilu(A1))

    if info:
        raise RuntimeError(info)
    else:
        return duvp


bfs.lu0 = splu(condense(bfs.S, I=bfs.I).T)
bfs.__class__.creeping = creeping
bfs.__class__.jacobian_solver = jacobian_solver


if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    try:
        natural(bfs, bfs.creeping(), 0.,
                partial(callback, name=splitext(argv[0])[0]),
                lambda_stepsize0=50.,
                lambda_stepsize_max=150.,
                newton_tol=1e-9)
    except RangeException:
        print(f'Reynolds number sweep complete: {re}.')
