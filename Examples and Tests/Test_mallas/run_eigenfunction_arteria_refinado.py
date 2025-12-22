import sys
import numpy as np
from skfem import (
    Basis, ElementTriP2, ElementTriP1, ElementVector,
    asm, bmat, condense, BilinearForm, LinearForm, Mesh, MeshTri
)
from skfem.models.general import divergence
from skfem.models.poisson import vector_laplace
from skfem.helpers import dot, read_meshh5
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import h5py

# Importar malla
mesh_normal, dofs_boundary, _ = read_meshh5('mallas/Carotid_h1.h5', dim=2)
basis_p1 = Basis(mesh_normal, ElementTriP1())
boundary_wall    = dofs_boundary[0]
boundary_inflow  = dofs_boundary[1]
boundary_outflow = dofs_boundary[2:-1]

# Refinar malla
mesh = mesh_normal.refined(1)

# Buscar nodos de frontera refinados
coords_inflow  =  mesh.p[:, basis_p1.get_dofs(boundary_inflow).all()]
coords_outflow = [mesh.p[:, basis_p1.get_dofs(dof).all()] for dof in boundary_outflow]

def find_new_boundary(coords, mesh_ref, tol=1e-3):
    x, y = coords
    pol1 = np.polyfit(x, y, deg=2)
    x_refined, y_refined = mesh_ref.p
    mask_domain = (x_refined >= np.min(x)) & (x_refined <= np.max(x))
    y_recta     = np.polyval(pol1, x_refined)
    mask        = (np.abs(y_refined - y_recta) < tol) & mask_domain
    return np.where(mask)[0], x_refined[mask], y_refined[mask]

dofs_inflow_refined, x_inflow_refined, y_inflow_refined = find_new_boundary(coords_inflow, mesh, tol=1e-3)
dofs_outflow_refined = []
for coords in coords_outflow:
    dofs_out, x_out, y_out = find_new_boundary(coords, mesh, tol=1e-3)
    dofs_outflow_refined.append(dofs_out)
dofs_wall_refined = np.setdiff1d(mesh.boundary_nodes(), np.concatenate([dofs_inflow_refined] + dofs_outflow_refined))

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
M_p = asm(mass_matrix, basis_p)

# Construcción del ssitema
K = bmat([[nu*A,    B.T],
            [B   ,    None]], format='csr') 

zeros = sp.csr_matrix((basis_p.N, basis_p.N))

L = bmat([[M,    None],
          [None, zeros]], format='csr')

# Condiciones de frontera
D_u    = np.array([[2*i, 2*i+1] for i in dofs_wall_refined]).flatten()
dofs_p = np.array([[2*i, 2*i+1] for i in dofs_outflow_refined[0]]).flatten()
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

# Exportar a CSV
np.savez_compressed(
    'eigenmodes_arterias_2D_norm_refinado.npz',
    velocity=all_u_velocity,
    pressure=all_p_pressure,
    eigenvalues=eigenvalues
)

print('Done!')