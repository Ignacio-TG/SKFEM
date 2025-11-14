import numpy as np
from skfem import (
    MeshTri, Basis, ElementTriP2, ElementTriP1, ElementVector,
    asm, bmat, condense, solve
)
from skfem.assembly import LinearForm
from skfem.models.general import divergence
from skfem.models.poisson import vector_laplace

# Perfiles exacto
def u_exact_x(x, y):
    nu   = 0.1
    Re   = 1/nu
    lamb = Re/2-np.sqrt(Re*Re/4+4*np.pi*np.pi)
    return 1- np.exp(lamb*x)*np.cos(2*np.pi*y)

def u_exact_y(x, y):
    nu   = 0.1
    Re   = 1/nu
    lamb = Re/2-np.sqrt(Re*Re/4+4*np.pi*np.pi)
    return  lamb/(2*np.pi)*np.exp(lamb*x)*np.sin(2*np.pi*y)

def p_exact_xy(x, y):
    nu   = 0.1
    Re   = 1/nu
    lamb = Re/2-np.sqrt(Re*Re/4+4*np.pi*np.pi)
    return 0.5*np.exp(2*lamb*x) + 0*y

def f_exact_x(x, y):
    nu   = 0.1
    Re   = 1/nu
    lamb = Re/2-np.sqrt(Re*Re/4+4*np.pi*np.pi)
    return (nu * (lamb**2 - 4*np.pi**2) * np.exp(lamb*x) * np.cos(2*np.pi*y)
            + lamb * np.exp(2*lamb*x))

def f_exact_y(x, y):
    nu   = 0.1
    Re   = 1/nu
    lamb = Re/2-np.sqrt(Re*Re/4+4*np.pi*np.pi)
    return (-nu * lamb * (lamb**2 - 4*np.pi**2) / (2*np.pi)
            * np.exp(lamb*x) * np.sin(2*np.pi*y))


def build_and_solve_stokes(nx=8, ny=8):
    # Definir dominio y mallado
    mesh = MeshTri.init_tensor(
        np.linspace(0.0, 1.0, nx + 1), 
        np.linspace(0.0, 1.0, ny + 1)
    )
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
        'u': Basis(mesh, element['u'], intorder=3),
        'p': Basis(mesh, element['p'], intorder=3),
    }

    basis_u = basis['u']
    basis_p = basis['p']
    Nu      = basis_u.N
    Np      = basis_p.N
    nu      = 0.1  # Viscosidad cinemática
    # Ensamblaje de matrices
    A =  nu *asm(vector_laplace, basis['u'])
    B = -asm(divergence, basis['u'], basis['p'])

    K = bmat([[A, B.T],
            [B, None]], format='csr')

    x_boundary = np.zeros(Nu + Np)

    # Fronteras de la velocidad

    # Obtener DOFs de las fronteras de la base de velocidad
    dofs_left     = basis_u.get_dofs('left').all()
    dofs_right    = basis_u.get_dofs('right').all()
    dofs_top      = basis_u.get_dofs('top').all()
    dofs_bottom   = basis_u.get_dofs('bottom').all()

    # Elementos con condiciones de Dirichlet para la velocidad
    # Impares y, pares x
    y_left   = basis_u.doflocs[1, dofs_left[1::2]] 
    y_right  = basis_u.doflocs[1, dofs_right[1::2]]

    x_left   = basis_u.doflocs[0, dofs_left[::2]]  
    x_right  = basis_u.doflocs[0, dofs_right[::2]]

    x_top    = basis_u.doflocs[0, dofs_top[::2]]
    x_bottom = basis_u.doflocs[0, dofs_bottom[::2]]

    y_top    = basis_u.doflocs[1, dofs_top[1::2]]
    y_bottom = basis_u.doflocs[1, dofs_bottom[1::2]]


    x_boundary[dofs_left[::2]]    =  u_exact_x(x_left, y_left)  # componente x
    x_boundary[dofs_left[1::2]]   =  u_exact_y(x_left, y_left)  # componente y

    x_boundary[dofs_right[::2]]   =  u_exact_x(x_right, y_right)  # componente x
    x_boundary[dofs_right[1::2]]  =  u_exact_y(x_right, y_right)  # componente y

    x_boundary[dofs_top[::2]]     =  u_exact_x(x_top, y_top)  # componente x
    x_boundary[dofs_top[1::2]]    =  u_exact_y(x_top, y_top)  # componente y

    x_boundary[dofs_bottom[::2]]  =  u_exact_x(x_bottom, y_bottom)  # componente x
    x_boundary[dofs_bottom[1::2]] =  u_exact_y(x_bottom, y_bottom)  # componente y

    dofs_u_boundary = np.concatenate([
        dofs_left,
        dofs_right,
        dofs_top,
        dofs_bottom
    ])

    # Obtener DOFs de las fronteras de la base de velocidad
    dofs_left_p     = basis_p.get_dofs('left').all()
    dofs_right_p    = basis_p.get_dofs('right').all()
    dofs_top_p      = basis_p.get_dofs('top').all()
    dofs_bottom_p   = basis_p.get_dofs('bottom').all()

    # x_boundary[Nu + dofs_left_p]    = p_exact_xy(basis_p.doflocs[0, dofs_left_p]  , basis_p.doflocs[1, dofs_left_p]  )
    # x_boundary[Nu + dofs_right_p]   = p_exact_xy(basis_p.doflocs[0, dofs_right_p] , basis_p.doflocs[1, dofs_right_p] )
    # x_boundary[Nu + dofs_top_p]     = p_exact_xy(basis_p.doflocs[0, dofs_top_p]   , basis_p.doflocs[1, dofs_top_p]   )
    # x_boundary[Nu + dofs_bottom_p]  = p_exact_xy(basis_p.doflocs[0, dofs_bottom_p], basis_p.doflocs[1, dofs_bottom_p])
    x_boundary[Nu + 0]  = 0.5

    dofs_p_boundary = Nu + np.concatenate([
        np.array([0])
        # dofs_right_p,
        # dofs_top_p,
        # dofs_bottom_p
    ])

    D_all = np.unique(np.concatenate([dofs_u_boundary, dofs_p_boundary]))

    @LinearForm
    def rhs_u(v, w):
        x, y = w.x
        fx = f_exact_x(x, y)
        fy = f_exact_y(x, y)
        return fx * v[0] + fy * v[1]

    b_u = asm(rhs_u, basis_u) 
    b_p = np.zeros(Np)  
    F   = np.concatenate([b_u, b_p])

    # Resolver sistema condensado
    sol_full = solve(*condense(K, F, D=D_all, x=x_boundary))

    # Extraer soluciones
    u_sol = sol_full[:Nu]
    p_sol = sol_full[Nu:]

    return u_sol, p_sol, basis_u, basis_p