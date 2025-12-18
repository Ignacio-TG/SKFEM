import numpy as np
import pandas as pd
from skfem import (
    Basis, ElementVector, asm, bmat, condense, BilinearForm, ElementTetP2, ElementTetP1
)
from skfem.models.general import divergence
from skfem.models.poisson import vector_laplace
from skfem.helpers import dot, read_meshh5

import scipy.sparse as sp
from scipy.sparse.linalg import eigs

import pandas as pd


mesh, boundaries_dofs, boundaries_elem = read_meshh5('mallas/Carotid3D_h1.h5', dim=3)
boundary_wall    = boundaries_dofs[0]
boundary_outflow = boundaries_dofs[2]

# Definir elementos y bases (P2 para velocidad, P1 para presión) en 3D
element = {
    'u': ElementVector(ElementTetP2(), dim=3),
    'p': ElementTetP1(),
}
basis = {
    'u': Basis(mesh, element['u'], intorder=4),
    'p': Basis(mesh, element['p'], intorder=4),
}
basis_u, basis_p = basis['u'], basis['p']
Nu, Np = basis_u.N, basis_p.N
N      = Nu + Np


# Ensamblaje de matrices
@BilinearForm
def mass_matrix(u, v, w):
    return dot(u, v)

nu = 1.0

A =  asm(vector_laplace, basis_u)               
B = -asm(divergence, basis_u, basis_p)   
M =  asm(mass_matrix, basis_u)  
M_p = asm(mass_matrix, basis_p)


# Construcción del ssitema
K = bmat([[nu*A,    B.T],
            [B   ,    None]], format='csr') 

zeros = sp.csr_matrix((basis_p.N, basis_p.N))

L = bmat([[M,    None],
          [None, zeros]], format='csr')

# Condiciones de frontera
D_u    = np.unique(basis_u.get_dofs(boundary_wall).all())
dofs_p = basis_p.get_dofs(boundary_outflow).all()
D_all  = np.concatenate([D_u, np.array([dofs_p[0]]) + Nu])

# Remover DOFs
A_sys, xI, I   = condense(K, D=D_all)
M_sys, xIM, IM = condense(L, D=D_all)

# Resolver problema de valores propios generalizado
vals, vecs = eigs(A_sys, k=40, M=M_sys, sigma=0.0, which='LM', OPpart='r')

# Filtrar valores propios reales (parte imaginaria exactamente cero)
mask_real = np.isclose(vals.imag, 0)
vals = vals[mask_real].real
vecs = vecs[:, mask_real].real

# Ordenar autovalores
idx          = np.argsort(vals)
eigenvalues  = vals[idx]
eigenvectors = vecs[:, idx]

# Exportar todas las funciones propias y valores propios a CSV

# Crear arrays para todas las soluciones
n_modes        = len(eigenvalues)
all_u_velocity = np.zeros((Nu, n_modes))
all_p_pressure = np.zeros((Np, n_modes))

for sol_idx in range(n_modes):
    # Insertar la solución reducida en el vector global (con ceros en frontera)
    u_sol = np.zeros(K.shape[0])
    u_sol[I] = eigenvectors[:, sol_idx]

    u_norm = np.sqrt(u_sol[:Nu].T @ M @ u_sol[:Nu])
    p_norm = np.sqrt(u_sol[Nu:Nu+Np].T @ M_p @ u_sol[Nu:Nu+Np])
    
    # Separar componentes de velocidad y presión
    all_u_velocity[:, sol_idx] = u_sol[:Nu]/u_norm
    all_p_pressure[:, sol_idx] = u_sol[Nu:Nu+Np]/p_norm

# Exportar a npz
np.savez_compressed(
    'eigenmodes_arterias_3D_norm.npz',
    velocity_eigenfunctions=all_u_velocity.astype(np.float32),
    pressure_eigenfunctions=all_p_pressure.astype(np.float32),
    eigenvalues=eigenvalues.astype(np.float32)
)

print('Done!')