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

# Setear condición para la presión
def get_p_boundary_option():
    for arg in sys.argv[1:]:
        if arg.startswith("P_option="):
            try:
                return int(arg.split("=")[1])
            except ValueError:
                pass
    return 1 

P_boundary_option = get_p_boundary_option() # 1: presión media cero, 2: presión en un punto

# Importar malla
# Definir dominio y mallado
nx, ny = 32, 32
mesh = MeshTri.init_tensor(
    np.linspace(0.0, 1.0, nx + 1), 
    np.linspace(0.0, 1.0, ny + 1)
)
# Asignar ID a las fronteras
mesh = mesh.with_boundaries({
    'left':   lambda x: x[0] == 0.0,
    'right':  lambda x: x[0] == 1.0,
    'bottom': lambda x: x[1] == 0.0,
    'top':    lambda x: x[1] == 1.0,
})


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


if P_boundary_option == 1:
    print('Condición: presión media cero')
    @LinearForm
    def mean_vec(q, w):
        return q

    m    = asm(mean_vec, basis_p)    # shape: (Np,)
    Mcol = m.reshape((-1, 1))     # (Np,1)
    Mrow = m.reshape((1,  -1))    # (1,Np)

 
    # Construcción del ssitema
    K = bmat([[nu*A,    B.T, None],
            [B,    None, Mcol],
            [None, Mrow, np.array([1])]], format='csr') 

    zeros = sp.csr_matrix((basis_p.N, basis_p.N))  
    L = bmat([[M,   None, None],
            [None, zeros, Mcol*0],
            [None, Mrow*0, None]], format='csr')

    # Condiciones de frontera
    D_all = np.unique(basis_u.get_dofs().all())

elif P_boundary_option == 2:
    print('Condición: presión en un punto')

    # Construcción del ssitema
    K = bmat([[nu*A,    B.T],
              [B   ,    None]], format='csr') 
    
    zeros = sp.csr_matrix((basis_p.N, basis_p.N))  
    L = bmat([[M,    None],
              [None, zeros]], format='csr')
    
    # Condiciones de frontera
    D_all = np.unique(basis_u.get_dofs().all())

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
pd.DataFrame(all_u_velocity).to_csv('Data/velocity_eigenfunctions_stokes_sincos.csv', index=False)
pd.DataFrame(all_p_pressure).to_csv('Data/pressure_eigenfunctions_stokes_sincos.csv', index=False)
pd.DataFrame({'eigenvalue': eigenvalues}).to_csv('Data/eigenvalues_stokes_sincos.csv', index=False)

print('Done!')