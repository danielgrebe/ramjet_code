import numpy as np
from ambiance import Atmosphere
from pygasflow import isentropic_solver, shockwave_solver, rayleigh_solver
import pygasflow.isentropic as ise
import pygasflow.shockwave as shock
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm

MACH_FIRST_CHOC = 2.3
SECOND_RAMP = 10  # degree
MACH_DEBUT_COMBUSTION = 0.5
MACH_FIN_COMBUSTION = 0.8
GAMMA_POST_COMBUSTION = 1.32
MOL_MASS_POST_COMBUSTION = 28.917
GAS_CONSTANT = 8314.46261815324
R_AIR = GAS_CONSTANT / 28.97
DESIGN_THRUST = 10e3  # N
JET_WIDTH = 0.7  # m
C_p_gas = 1.006e3  # J/kgK
GRAVITY = 9.81  # Kg/N
FUEL_ENERGY = 42.44e6  # J/kg


def diffuser(mach):
    # first shock
    T2_T1 = ise.temperature_ratio(MACH_FIRST_CHOC)/ise.temperature_ratio(mach)
    mn1 = shock.get_upstream_normal_mach_from_ratio('temperature', T2_T1)
    beta = np.rad2deg(np.arcsin(mn1/mach))
    choc1 = shockwave_solver('m1', mach, 'beta', beta, to_dict=True)
    # print(choc1['m2'])
    # print(choc1['theta'])
    
    # second shock
    choc2 = shockwave_solver('m1', choc1['m2'], 'theta', SECOND_RAMP, 
                to_dict=True)
    # print(choc2['m2'])

    # isentropique
    P02_P01 = choc1['tpr']*choc2['tpr']

    return P02_P01

def design_point(M3=MACH_DEBUT_COMBUSTION, M4=MACH_FIN_COMBUSTION):
    altitude = 20000
    mach = 2.8
    # initial stagnation conditions
    atm_cond = Atmosphere(altitude)
    P1 = atm_cond.pressure  # Pa
    T1 = atm_cond.temperature  # K
    rho1 = atm_cond.density  # kg/m**3
    c1 = atm_cond.speed_of_sound  # m/s
    U1 = c1 * mach
    # print([P1, T1, rho1, c1])
    
    ratios1 = isentropic_solver('m', mach, to_dict=True)
    P01 = P1/ratios1['pr']
    T01 = T1/ratios1['tr']
    rho01 = rho1/ratios1['dr']
    # print([P01, T01, rho01])
    
    # conditions after diffuseur
    P02_P01 = diffuser(mach)
    # print(P02_P01)
    P02 = P01 * P02_P01
    rho02 = P02 / (R_AIR * T01)
    
    
    # conditions debut combustion
    # M3 = MACH_DEBUT_COMBUSTION
    # M4 = MACH_FIN_COMBUSTION
    T03 = T01  # car adiabatique
    P03 = P02  # car isentropique apres diffuseur
    start_comb = isentropic_solver('m', M3, to_dict=True)
    T3 = T03 * start_comb['tr']
    P3 = P03 * start_comb['pr']
    print((T3, P3))
    
    
    
    # conditions after combustion
    rayleigh3 = rayleigh_solver('m', M3, to_dict=True, 
                    gamma=GAMMA_POST_COMBUSTION)
    rayleigh4 = rayleigh_solver('m', M4, to_dict=True, 
                    gamma=GAMMA_POST_COMBUSTION)
    T04_T03 = rayleigh4['ttrs']/rayleigh3['ttrs']
    P04_P03 = rayleigh4['tprs']/rayleigh3['tprs']
    # print(T04_T03)
    # print(P04_P03)
    T04 = T03 * T04_T03
    P04 = P03 * P04_P03
    # print(T04)
    # print(P04)
    R_combustion = GAS_CONSTANT / MOL_MASS_POST_COMBUSTION
    rho04 = P04 / (R_combustion * T04)
    # print(rho04)
    
    # Conditions au col
    throat = isentropic_solver('m', 1, to_dict=True, 
                    gamma=GAMMA_POST_COMBUSTION)
    T_star = throat['tr'] * T04
    rho_star = throat['dr'] * rho04
    c_star = np.sqrt(GAMMA_POST_COMBUSTION * T_star * R_combustion)
    # print([c_star, rho_star])

    # Conditions à la sortie
    P6 = atm_cond.pressure
    P6_P04 = P6 / P04
    exit = isentropic_solver('pressure', P6_P04, to_dict=True,
                gamma=GAMMA_POST_COMBUSTION)
    M6 = exit['m']
    T6 = exit['tr'] * T04
    A_Astar = exit['ars']
    C6 = np.sqrt(GAMMA_POST_COMBUSTION*R_combustion*T6)
    U6 = exit['urs'] * c_star
    U6 = C6 * M6
    # print([M6, U6, U1])
    
    m_dot = DESIGN_THRUST / (U6 - U1)
    # print(m_dot)
    A_star = m_dot / (c_star * rho_star)
    h_star = A_star / JET_WIDTH
    A_exit = A_star * A_Astar

    # print([A_star, h_star])

    # Fuel consumption
    q = C_p_gas * (T04 - T03)
    fuel_ratio = q / FUEL_ENERGY
    # print(fuel_ratio)
    fuel_flow = m_dot * fuel_ratio
    # print(fuel_flow)
    

    thrust = DESIGN_THRUST
    isp = thrust / GRAVITY / fuel_flow
    # print(isp)

    # Section chambre de combustion rho03 = rho02
    rho03 = rho02
    chambre_in = isentropic_solver('m', M3, to_dict=True)
    rho3 = rho03 * chambre_in['dr']
    T3 = T03 * chambre_in['tr']
    c3 = np.sqrt(1.4 * R_AIR * T3)
    A_chambre = m_dot / c3 / rho3
    h_chambre = A_chambre / JET_WIDTH


    # hauteur diffuseur
    diff_throat = isentropic_solver('m', 1, to_dict=True, 
                    gamma=1.4)
    T2_star = diff_throat['tr']*T01
    rho2_star = diff_throat['dr']*rho02
    c2_star = np.sqrt(R_AIR * 1.4 * T2_star)
    A_throat_diff = m_dot / c2_star / rho2_star
    h_throat_diff = A_throat_diff / JET_WIDTH
    return (isp, h_star, h_throat_diff, h_chambre, fuel_flow, fuel_ratio,
        A_exit)

def off_design_throat_fixe(altitude, 
                           mach, 
                           h_col,
                           A_exit,
                           M3=MACH_DEBUT_COMBUSTION,
                           M4=MACH_FIN_COMBUSTION):
    # initial stagnation conditions
    atm_cond = Atmosphere(altitude)
    P1 = atm_cond.pressure  # Pa
    T1 = atm_cond.temperature  # K
    rho1 = atm_cond.density  # kg/m**3
    c1 = atm_cond.speed_of_sound  # m/s
    U1 = c1 * mach
    # print([P1, T1, rho1, c1])
    
    ratios1 = isentropic_solver('m', mach, to_dict=True)
    P01 = P1/ratios1['pr']
    T01 = T1/ratios1['tr']
    rho01 = rho1/ratios1['dr']
    # print([P01, T01, rho01])
    
    # conditions after diffuseur
    P02_P01 = diffuser(mach)
    # print(P02_P01)
    P02 = P01 * P02_P01
    rho02 = P02 / (R_AIR * T01)
    
    
    # conditions debut combustion
    # M3 = MACH_DEBUT_COMBUSTION
    # M4 = MACH_FIN_COMBUSTION
    T03 = T01  # car adiabatique
    P03 = P02  # car isentropique apres diffuseur

    # conditions after combustion
    rayleigh3 = rayleigh_solver('m', M3, to_dict=True, 
                    gamma=GAMMA_POST_COMBUSTION)
    rayleigh4 = rayleigh_solver('m', M4, to_dict=True, 
                    gamma=GAMMA_POST_COMBUSTION)
    T04_T03 = rayleigh4['ttrs']/rayleigh3['ttrs']
    P04_P03 = rayleigh4['tprs']/rayleigh3['tprs']
    # print(T04_T03)
    # print(P04_P03)
    T04 = T03 * T04_T03
    P04 = P03 * P04_P03
    # print(T04)
    # print(P04)
    R_combustion = GAS_CONSTANT / MOL_MASS_POST_COMBUSTION
    rho04 = P04 / (R_combustion * T04)
    # print(rho04)
    
    # Conditions au col
    throat = isentropic_solver('m', 1, to_dict=True, 
                    gamma=GAMMA_POST_COMBUSTION)
    T_star = throat['tr'] * T04
    rho_star = throat['dr'] * rho04
    c_star = np.sqrt(GAMMA_POST_COMBUSTION * T_star * R_combustion)
    # print([c_star, rho_star])

    # Conditions à la sortie
    A_star = h_col * JET_WIDTH
    A_Astar = A_exit / A_star
    exit = isentropic_solver('crit_area_super', A_Astar, to_dict=True,
                gamma=GAMMA_POST_COMBUSTION)
    M6 = exit['m']
    T6 = exit['tr'] * T04
    P6 = exit['pr'] * P04
    C6 = np.sqrt(GAMMA_POST_COMBUSTION*R_combustion*T6)
    U6 = exit['urs'] * c_star
    U6 = C6 * M6
    # print([M6, U6, U1])
    
    P_atm = atm_cond.pressure
    m_dot = A_star * c_star * rho_star
    # print(m_dot)
    # print(P6 - P_atm)
    thrust = (U6 - U1) * m_dot + A_exit * (P6 - P_atm)

    # print([A_star, h_star])

    # Fuel consumption
    q = C_p_gas * (T04 - T03)
    fuel_ratio = q / FUEL_ENERGY
    # print(fuel_ratio)
    fuel_flow = m_dot * fuel_ratio
    # print(fuel_flow)
    

    isp = thrust / GRAVITY / fuel_flow
    # print(isp)

    # hauteur diffuseur
    diff_throat = isentropic_solver('m', 1, to_dict=True, 
                    gamma=1.4)
    T2_star = diff_throat['tr']*T01
    # print(T2_star)
    rho2_star = diff_throat['dr']*rho02
    c2_star = np.sqrt(R_AIR * 1.4 * T2_star)
    A_throat_diff = m_dot / c2_star / rho2_star
    h_throat_diff = A_throat_diff / JET_WIDTH
    # print(A_throat_diff)
    # print(h_throat_diff)
    return thrust, isp, h_throat_diff, m_dot, fuel_ratio

def main():
    design_point()
    m3 = np.linspace(0.1, 0.6, 20)
    m4 = np.linspace(0.7, 0.9, 20)
    m3v, m4v = np.meshgrid(m3, m4)
    vfunc = np.vectorize(design_point)
    isp, h_star, h_diff, h_chambre, *_ = vfunc(m3v, m4v)

    fig, (ax0, ax1) = plt.subplots(1,2,subplot_kw={"projection": "3d"})
    surf = ax0.plot_surface(m3v, m4v, isp, cmap=cm.coolwarm,
                    linewidth=0)
    ax0.set_xlabel('$M_3$')
    ax0.set_ylabel('$M_4$')
    ax0.set_zlabel('isp (s)')
    surf = ax1.plot_surface(m3v, m4v, h_chambre, cmap=cm.coolwarm,
                    linewidth=0)
    # ax1.set_zscale('log')
    ax1.set_xlabel('$M_3$')
    ax1.set_ylabel('$M_4$')
    ax1.set_zlabel('hauteur du col (m)')
    print(design_point(0.3, 0.6))
    _, h_col, _, _, _, _, A_exit = design_point()

    vfunc_off = np.vectorize(off_design_throat_fixe)
    alt = np.linspace(15000, 25000, 20)
    mach = np.linspace(2.5, 3.0, 20)
    altv, machv = np.meshgrid(alt, mach)
    
    off_thrust, off_isp, off_h_diff, off_m_dot, fuel_frac = vfunc_off(altv, machv, h_col,
                A_exit)
   

    fig1, ((ax2, ax3),(ax4, ax5)) = plt.subplots(2,2,subplot_kw={"projection": "3d"})
    surf = ax2.plot_surface(altv, machv, off_isp, cmap=cm.coolwarm,
                    linewidth=0)
    ax2.set_xlabel('$Altitude (m)$')
    ax2.set_ylabel('$Mach$')
    ax2.set_zlabel('isp (s)')
    surf = ax3.plot_surface(altv, machv, off_thrust, cmap=cm.coolwarm,
                    linewidth=0)
    #ax3.set_zscale('log')
    ax3.set_xlabel('$Altitude (m)$')
    ax3.set_ylabel('$Mach$')
    ax3.set_zlabel('Poussée (N)')
    surf = ax4.plot_surface(altv, machv, fuel_frac, cmap=cm.coolwarm,
                    linewidth=0)
    ax4.set_xlabel('$Altitude (m)$')
    ax4.set_ylabel('$Mach$')
    ax4.set_zlabel('Ratio carburant')
    surf = ax5.plot_surface(altv, machv, off_m_dot, cmap=cm.coolwarm,
                    linewidth=0)
    #ax3.set_zscale('log')
    ax5.set_xlabel('$Altitude (m)$')
    ax5.set_ylabel('$Mach$')
    ax5.set_zlabel('Débit massique (kg/s)')
    print(off_design_throat_fixe(20000, 2.8, h_col, A_exit))
    # plt.figure()
    # plt.scatter(altv, off_thrust)
    plt.show()

if __name__ == '__main__':
    main()
    
    

