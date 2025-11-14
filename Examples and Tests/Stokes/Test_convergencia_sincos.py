import numpy as np
from skfem import Functional
import pandas as pd
from build_sincos import build_and_solve_stokes, u_exact_x, u_exact_y, p_exact_xy, f_exact_x, f_exact_y

# Definición de normas
nu   = 0.1
Re   = 1/nu
lamb = Re/2-np.sqrt(Re*Re/4+4*np.pi*np.pi)

@Functional
def err_L2_p(w):
    x, y = w.x
    p_e = p_exact_xy(x, y)
    p_h = w["p"]
    return (p_e - p_h)**2

@Functional
def err_H1_semi_p(w):
    x, y = w.x

    # gradiente exacto de la presion
    dp_dx_e = 2*np.pi * np.cos(2*np.pi*x) * np.sin(2*np.pi*y)
    dp_dy_e = 2*np.pi * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)

    # gradiente FEM de la presion: vector 2D
    gp = w["p"].grad
    dp_dx_h = gp[0]
    dp_dy_h = gp[1]

    return ((dp_dx_e - dp_dx_h)**2
          + (dp_dy_e - dp_dy_h)**2)

@Functional
def err_L2_u(w):
    x, y = w.x

    # solucion exacta
    ux_e = u_exact_x(x, y)
    uy_e = u_exact_y(x, y)

    # solucion FEM reconstruida
    ux_h = w["u"][0]
    uy_h = w["u"][1]

    return (ux_e - ux_h)**2 + (uy_e - uy_h)**2

@Functional
def err_H1_semi_u(w):
    x, y = w.x

    # derivadas exactas de Kovasznay
    dux_dx_e =  4*np.pi**2 * np.cos(2*np.pi*x) * np.cos(2*np.pi*y)
    dux_dy_e = -4*np.pi**2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)

    duy_dx_e = 4*np.pi**2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    duy_dy_e = -4*np.pi**2 * np.cos(2*np.pi*x) * np.cos(2*np.pi*y)

    # gradiente FEM
    gu = w["u"].grad
    dux_dx_h = gu[0, 0]
    dux_dy_h = gu[0, 1]
    duy_dx_h = gu[1, 0]
    duy_dy_h = gu[1, 1]

    return ((dux_dx_e - dux_dx_h)**2
          + (dux_dy_e - dux_dy_h)**2
          + (duy_dx_e - duy_dx_h)**2
          + (duy_dy_e - duy_dy_h)**2)


# Prueba de convergencia

N = [2, 4, 8, 16, 32, 64, 128, 256]
err_u_L2      = []
err_u_H1_semi = []
err_p_L2      = []
err_p_H1_semi = []

for n in N:
    u_sol, p_sol, basis_u, basis_p = build_and_solve_stokes(nx=n, ny=n)

    # Calcular errores
    L2_u_sq      = err_L2_u.assemble(basis_u, u=u_sol)
    H1_semi_u_sq = err_H1_semi_u.assemble(basis_u, u=u_sol)

    L2_p_sq      = err_L2_p.assemble(basis_p, p=p_sol)
    H1_semi_p_sq = err_H1_semi_p.assemble(basis_p, p=p_sol)

    # Almacenar errores
    err_u_L2.append(np.sqrt(L2_u_sq))
    err_u_H1_semi.append(np.sqrt(H1_semi_u_sq))
    err_p_L2.append(np.sqrt(L2_p_sq))
    err_p_H1_semi.append(np.sqrt(H1_semi_p_sq))
    print(f"n={n}: ||u-u_h||_L2={err_u_L2[-1]:.3e}, ||u-u_h||_H1_semi={err_u_H1_semi[-1]:.3e}, "
          f"||p-p_h||_L2={err_p_L2[-1]:.3e}, ||p-p_h||_H1_semi={err_p_H1_semi[-1]:.3e}")
    
# Crear DataFrame y guardar en CSV
data = {
    'N': N,
    'err_u_L2': err_u_L2,
    'err_u_H1_semi': err_u_H1_semi,
    'err_p_L2': err_p_L2,
    'err_p_H1_semi': err_p_H1_semi,
}
df = pd.DataFrame(data)
df.to_csv('convergencia_stokes_sincos.csv', index=False)