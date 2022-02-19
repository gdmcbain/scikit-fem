import pytest

import numpy as np

from skfem import (
    Mesh,
    MeshTet,
    ElementHcurl,
    ElementTetN0,
    asm,
    solve,
    condense,
)
from skfem.assembly.basis.cell_basis import CellBasis

from docs.examples.ex33 import dudv, f, fv


@pytest.mark.parametrize("e,m", [(ElementTetN0(), MeshTet())])
def test_convergence_nedelec0(e: ElementHcurl, m: Mesh, Nitrs: int = 3):

    L2s = np.zeros(Nitrs)
    Hcurls = np.zeros(Nitrs)
    hs = np.zeros(Nitrs)

    for itr in range(Nitrs):
        b = CellBasis(m, e)
        x = solve(*condense(dudv.assemble(b), fv.assemble(b), D=b.get_dofs()))

        m = m.refined()
        pass
