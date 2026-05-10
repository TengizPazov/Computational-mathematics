import numpy as np
import matplotlib.pyplot as plt
from params import *

def HLLC(U_L, U_R):
    U_L, U_R = np.array(U_L), np.array(U_R)
    ro_L, ro_R = U_L[0], U_R[0]
    u_L, u_R = U_L[1] / ro_L, U_R[1] / ro_R
    p_L = (gamma - 1) * (U_L[4] - 0.5 * U_L[0] * (u_L)**2)
    p_R = (gamma - 1) * (U_R[4] - 0.5 * U_R[0] * (u_R)**2)
    a_L, a_R = np.sqrt(gamma * p_L / ro_L), np.sqrt(gamma * p_R / ro_R)

    #вычисляем S_L  и S_R
    S_L = np.min([u_L - a_L, u_R - a_R], axis=0)
    S_R = np.max([u_L + a_L, u_R + a_R], axis=0)

    S_star = (p_R - p_L + ro_L * u_L * (S_L - u_L) - ro_R * u_R * (S_R - u_R)) / (ro_L * (S_L - u_L) - ro_R * (S_R - u_R))
    #вычисляем F_L и F_R
    F_L = np.array([ro_L * u_L, ro_L * u_L**2 + p_L, U_L[2] * u_L, U_L[3] * u_L, u_L * (U_L[4] + p_L)])
    F_R = np.array([ro_R * u_R, ro_R * u_R**2 + p_R, U_R[2] * u_R, U_R[3] * u_R, u_R * (U_R[4] + p_R)])

    D_star = np.array([0, 1, 0, 0, S_star])
    F_L_star = (S_star * (S_L * U_L - F_L) + S_L * (p_L + ro_L * (S_L - u_L) * (S_star - u_L)) * D_star) / (S_L - S_star)
    F_R_star = (S_star * (S_R * U_R - F_R) + S_R * (p_R + ro_L * (S_R - u_R) * (S_star - u_R)) * D_star) / (S_R - S_star)

    if S_L >= 0:
        return F_L
    elif S_L <= 0 <= S_star:
        return F_L_star
    elif S_star <= 0 <= S_R:
        return F_R_star
    elif S_R <= 0:
        return F_R