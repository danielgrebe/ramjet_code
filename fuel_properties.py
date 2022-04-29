import cantera as ct
ct.suppress_thermo_warnings()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#phi = np.linspace(0.01,2,1000)
#gas = ct.Solution('gri30.xml')
#T_aft = []
#for i in range(0,len(phi)):
#    n_tot = 1 + (2/phi[i]) + (7.52/phi[i])
#    n_ch4 = 1/n_tot
#    n_o2 = (2/phi[i])/n_tot
#    n_n2 = (7.52/phi[i])/n_tot
#Gas Mixture
#    gas.TPX = 298.15, 101325, {'CH4':n_ch4,'O2':n_o2,'N2':n_n2}
#Equilibrum conditions
#    gas.equilibrate('UV','auto')
#    T_aft.append(gas.T)

#Plotting
#plt.plot(phi, T_aft, color='green')
#plt.xlabel('Equivalence ratio ($phi$)',fontsize=12)
#plt.ylabel('Température adiabatique de flamme [K]',fontsize=12)
#plt.grid()
#plt.title('Teméprature de flamme vs Equivalence Ratio \n for Methane-Air Combustion',fontsize=13,fontweight='bold')
#plt.show()
pass
#Questions combustion
#1: Stoechiométrique ? rapport des masses
#2: Propriétés à la sortie de la chambre de combustion (masse molaire et gamma)
#3: Apport énergétique
#Tout ca selon une plage de phi

REACTION_FILE = "JP10highT.yaml"
OUTPUT_BASEBAME = "JP10_stoich"

M3 = 0.3
if M3 == 0.3:
    INITIAL_TEMP = 546  # K
    INITIAL_PRESSURE = 132724 # Pa
else:
    INITIAL_TEMP = 529  # K
    INITIAL_PRESSURE = 119098 # Pa

RATIO_STOCH = 10+16/4
PHI_MIN = 0.01
PHI_MAX = 0.3
N = 50


def temp_adiabatique(phi):
    g = ct.Solution(REACTION_FILE)
    g.TPX = INITIAL_TEMP, INITIAL_PRESSURE, {'C10H16': 1,
                                       'O2': RATIO_STOCH / phi,
                                       'N2': RATIO_STOCH * 3.76 / phi}
    g.equilibrate('HP')
    return g.T

def enthalpy(phi):
    g = ct.Solution(REACTION_FILE)
    g.TPX = INITIAL_TEMP, INITIAL_PRESSURE, {'C10H16': 1,
                                       'O2': RATIO_STOCH / phi,
                                       'N2': RATIO_STOCH * 3.76 / phi}
    h_init = g.enthalpy_mass
    Y_JP10 = g["C10H16"].Y[0]
    g.equilibrate('HP')
    g.TP = INITIAL_TEMP, INITIAL_PRESSURE
    h_final = g.enthalpy_mass
    return (h_final-h_init)/Y_JP10/-1e6

def produits(phi):
    g = ct.Solution(REACTION_FILE)
    g.TPX = INITIAL_TEMP, INITIAL_PRESSURE, {'C10H16': 1,
                                       'O2': RATIO_STOCH / phi,
                                       'N2': RATIO_STOCH * 3.76 / phi}
    g.equilibrate('HP')
    return g


def main():
    phi_array = np.linspace(PHI_MIN, PHI_MAX, N)
    temp_array = np.array([temp_adiabatique(phi) for phi in phi_array])
    enthalpy_array = np.array([enthalpy(phi) for phi in phi_array])
    plt.plot(phi_array, temp_array)
    plt.xlabel(r'$\Phi$')
    plt.ylabel(r'Temperature de flamme (K)')
    plt.figure()
    plt.plot(phi_array, enthalpy_array)
    plt.xlabel(r'$\Phi$')
    plt.ylabel(r'LHV')
    phi_min = phi_array[np.argwhere(temp_array >= 800)[0]]
    frac_min = 1 / (RATIO_STOCH / phi_min + RATIO_STOCH * 3.76 / phi_min + 1)
    phi_max = phi_array[np.argwhere(temp_array >= 800)[-1]]
    frac_max = 1 / (RATIO_STOCH / phi_max + RATIO_STOCH * 3.76 / phi_max + 1)
    print("La température de flamme est supérieur à 800 K pour phi entre \n"
          "{phi_min:.2f} et {phi_max:.2f}, ce qui équivaut à des fractions \n"
          "molaires de {frac_min:.3f} et {frac_max:.3f}".format(phi_min=float(phi_min),
                                                                phi_max=float(phi_max),
                                                                frac_min=float(frac_min), frac_max=float(frac_max)))

    g = ct.Solution(REACTION_FILE)
    g.TPX = INITIAL_TEMP, INITIAL_PRESSURE, {'C10H16': 1/4.7,
                                       'O2': RATIO_STOCH,
                                       'N2': RATIO_STOCH * 3.76}
    sol = ct.SolutionArray(g, (1))
    sol.to_pandas().to_excel(OUTPUT_BASEBAME + ".init.xlsx")
    sol.equilibrate('HP')
    sol.to_pandas().to_excel(OUTPUT_BASEBAME + ".xlsx")
    g()
    plt.show()


if __name__ == "__main__":
    main()
