import sys
import numpy as np
from skfem import (
    Basis, ElementTriP2, ElementTriP1, ElementVector,
    asm, bmat, condense, BilinearForm, LinearForm, Mesh, MeshTri
)
from skfem.models.general import divergence
from skfem.models.poisson import vector_laplace
from skfem.helpers import dot
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import h5py

# Importar malla
f = h5py.File('Carotid_h1.h5', 'r')
coordinates_mesh = f['mesh/coordinates'][:]
elements_mesh    = f['mesh/topology'][:]

coordinates_boundaries = f['/boundaries/coordinates'][:]
elements_boundaries    = f['/boundaries/topology'][:]
values_boundaries      = f['/boundaries/values'][:]

boundary_1 = np.where(f['/boundaries/values'][:] == 1)[0]
boundary_elements_1 = elements_boundaries[boundary_1]

boundary_2 = np.where(f['/boundaries/values'][:] == 2)[0]
boundary_elements_2 = elements_boundaries[boundary_2]

boundary_3 = np.where(f['/boundaries/values'][:] == 3)[0]
boundary_elements_3 = elements_boundaries[boundary_3]

boundary_4 = np.where(f['/boundaries/values'][:] == 4)[0]
boundary_elements_4 = elements_boundaries[boundary_4]

boundary_5 = np.where(f['/boundaries/values'][:] == 5)[0]
boundary_elements_5 = elements_boundaries[boundary_5]

boundary_6 = np.where(f['/boundaries/values'][:] == 6)[0]
boundary_elements_6 = elements_boundaries[boundary_6]

boundary_7 = np.where(f['/boundaries/values'][:] == 7)[0]
boundary_elements_7 = elements_boundaries[boundary_7]

boundaries = [boundary_elements_1, boundary_elements_2,
             boundary_elements_3, boundary_elements_4,
              boundary_elements_5, boundary_elements_6,
              boundary_elements_7]

boundary_wall = boundary_1
boundary_inflow = boundary_2
boundary_outflow = np.concatenate([boundary_3, boundary_4, boundary_5, boundary_6, boundary_7])

mesh = MeshTri(coordinates_mesh.T, elements_mesh.T)

# Definir elementos y bases (P2 para velocidad, P1 para presión)
element = {
    'u': ElementVector(ElementTriP2()),
    'p': ElementTriP1(),
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


# Construcción del ssitema
K = bmat([[nu*A,    B.T],
            [B   ,    None]], format='csr') 

zeros = sp.csr_matrix((basis_p.N, basis_p.N))

L = bmat([[M,    None],
          [None, zeros]], format='csr')

# Condiciones de frontera
D_all = np.unique(basis_u.get_dofs(boundary_wall).all())

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
    
    # Separar componentes de velocidad y presión
    all_u_velocity[:, sol_idx] = u_sol[:Nu]
    all_p_pressure[:, sol_idx] = u_sol[Nu:Nu+Np]

# Exportar a CSV
pd.DataFrame(all_u_velocity).to_csv('eigenfunctions/velocity_eigenfunctions_arterias.csv', index=False)
pd.DataFrame(all_p_pressure).to_csv('eigenfunctions/pressure_eigenfunctions_arterias.csv', index=False)
pd.DataFrame({'eigenvalue': eigenvalues}).to_csv('eigenfunctions/eigenvalues_arterias.csv', index=False)

print('Done!')