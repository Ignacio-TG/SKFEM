import numpy as np
from skfem import (
    Basis, ElementTriP2, ElementTriP1, ElementVector,
    asm, bmat, condense, solve, Mesh
)
import pandas as pd
from skfem.models.general import divergence
from skfem.models.poisson import vector_laplace


# cargar malla
malla = Mesh.load("data/Ugeom.obj", force_meshio_type='triangle')
nodes  = malla.p
facets = malla.facets
boundary_facets = malla.boundary_facets()
boundary_nodes = malla.boundary_nodes()

#etiquetar fronteras
pared   = []
salida  = []
entrada = []
for i,e in enumerate(boundary_facets):
    facets_e = facets[:, e]
    nodox = nodes[0, facets_e]
    nodoy = nodes[1, facets_e]
    m = np.array([np.mean(nodox), np.mean(nodoy)])

    if m[1] > -2.0:
        pared.append(e)
    if m[1] <= -2.0 and m[0] > 0.0:
        salida.append(e)
    if m[1] <= -2.0 and m[0] < 0.0:
        entrada.append(e)

pared = np.array(pared)
salida = np.array(salida)
entrada = np.array(entrada)
coordenadas = nodes[:, boundary_nodes]

# Definir elementos y bases (P2 para velocidad, P1 para presión)
element = {
    'u': ElementVector(ElementTriP2()),
    'p': ElementTriP1(),
}
basis = {
    'u': Basis(malla, element['u'], intorder=4),
    'p': Basis(malla, element['p'], intorder=4),
}
basis_u, basis_p = basis['u'], basis['p']
Nu, Np = basis_u.N, basis_p.N
N      = Nu + Np

# Ensamblaje de matrices
A =  asm(vector_laplace, basis_u)               
B = -asm(divergence, basis_u, basis_p)   
nu = 0.035

K = bmat([[nu*A,    B.T],
          [B,    None]], format='csr') 

F_u = basis['u'].zeros()
F_p = basis['p'].zeros()
F = np.hstack([F_u, F_p])

# Condiciones de borde
def u_out_y(x, y):
    return  (x+0.02)*(0.5+x) + y*0.0

dofs_salida    = basis_u.get_dofs(entrada).all()
dofs_p_salida  = basis_p.get_dofs(salida).all()
xout   = basis_u.doflocs[0, dofs_salida[::2]] 
yout   = basis_u.doflocs[1, dofs_salida[1::2]]

x_boundaries = np.zeros(Nu+Np)
x_boundaries[dofs_salida[::2]]  = 0.0
x_boundaries[dofs_salida[1::2]] = -50*u_out_y(xout, yout)
x_boundaries[Nu + dofs_p_salida] = 0.0

D_all = np.concatenate([
    basis_u.get_dofs(pared).all(),
    basis_u.get_dofs(entrada).all(),
    Nu + dofs_p_salida
    # Nu+np.array([0])
]) 

sol = solve(*condense(K, F, D=D_all, x=x_boundaries))

u_sol = sol[:Nu]
p_sol = sol[Nu:Nu+Np]

# Exportar soluciones de referencia como csv
df_u = pd.DataFrame({
    'u_sol': u_sol,
})
df_p = pd.DataFrame({
    'p_sol': p_sol,
})
df_u.to_csv("data/reference_solution_velocity_stokes.csv", index=False)
df_p.to_csv("data/reference_solution_pressure_stokes.csv", index=False)