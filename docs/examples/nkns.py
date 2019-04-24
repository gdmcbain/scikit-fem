from ex27 import *

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
    A1, rhs1 = condense(step.S + reynolds *
                        block_diag([asm(acceleration_jacobian,
                                        step.basis['u'], w=u),
                                    csr_matrix((step.basis['p'].N,)*2)]),
                        rhs, duvp, I=step.I)
    duvp[step.I] = solve(A1, rhs1,
                         solver=solver_iter_krylov(
                             gmres,
                             build_pc_ilu(A1),
                             solve(step.lu0, rhs1,
                                   solver=lambda A, b: A.solve(b)),
                             tol=1e-12))
    return duvp


bfs.lu0 = splu(condense(bfs.S, I=bfs.I).T)
bfs.__class__.creeping = creeping
bfs.__class__.jacobian_solver = jacobian_solver


if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    milestones = [150., 450., 750.]
    natural(bfs, bfs.make_vector(), 0.,
            partial(callback,
                    name=splitext(argv[0])[0],
                    milestones=milestones),
            lambda_stepsize0=50.,
            milestones=milestones)
