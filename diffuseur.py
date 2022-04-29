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
MACH_FIN_COMBUSTION = 0.9
GAMMA_POST_COMBUSTION = 1.32
MOL_MASS_POST_COMBUSTION = 28.917
GAS_CONSTANT = 8314.46261815324
R_AIR = GAS_CONSTANT / 28.97
DESIGN_THRUST = 10e3  # N
JET_WIDTH = 1.0  # m
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
    theta = choc1['theta']
    
    # second shock
    choc2 = shockwave_solver('m1', choc1['m2'], 'theta', SECOND_RAMP, 
                to_dict=True)
    # print(choc2['m2'])

    # isentropique
    P02_P01 = choc1['tpr']*choc2['tpr']

    return P02_P01, theta

def main():
    pression, theta = diffuser(2.8)
    print("Point de design:")
    print("Rendement de pression: {:.4f}".format(pression))

    vdiffuser = np.vectorize(diffuser)
    mach = np.linspace(2.5, 3.0, 50)
    vpression, vtheta = vdiffuser(mach)

    print("Hors-design:")
    print("Mach 2.5: {:.4f} pressure recovery, {:.2f} ramp".format(
        vpression[0], vtheta[0]))
    print("Mach 3.0: {:.4f} pressure recovery, {:.2f} ramp".format(
        vpression[-1], vtheta[-1]))

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(mach, vpression, 'r-')
    ax2.plot(mach, vtheta, 'b-')
    
    ax1.set_xlabel('Nombre de Mach à l\'entrée')
    ax1.set_ylabel('Rendement de pression totale', color='r')
    ax2.set_ylabel('Angle de la première rampe', color='b')

    plt.show()

if __name__ == '__main__':
    main()
    
    

