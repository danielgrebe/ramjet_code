import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from scipy.optimize import fsolve
from iso_1D import Condensing_Ecoulement, IdealGasVariable, Geometry, InitialConditions
from numba import jit
mpl.rcParams['interactive'] = False

BASENAME = "data/test_nucleation_tuto"
GAMMA = 1.298
MASSE_MOLAIRE = 31.226
DEBIT_POWER_MIN = -9  # log10(kg/s)
DEBIT_POWER_MAX = 0  # log10(kg/s)
N_DEBIT = 50
N_POS = 10000
CRITERE_DEBUT = 0.14999


def main():
    geo = Geometry(BASENAME)
    init_cond = InitialConditions(2.033e5, 295)
    gas_prop = IdealGasVariable(frac_CO2_init=0.15)
    debits = [10 ** x for x in np.linspace(DEBIT_POWER_MIN, DEBIT_POWER_MAX, N_DEBIT)]
    ecoulements = [Condensing_Ecoulement(geo, init_cond, gas_prop, BASENAME, flow, n=1000) for flow in debits]
    x_points = np.linspace(0, geo.x_points[-1], N_POS)

    nuc_max = [max(ecoulement.nucleation_rate(x_points)) for ecoulement in ecoulements]
    debut_nuc = [x_points[np.argmax(ecoulement.frac_co2(x_points) < CRITERE_DEBUT)] for ecoulement in ecoulements]
    demi_nuc = [x_points[np.argmax(ecoulement.frac_co2(x_points) < CRITERE_DEBUT / 2)] for ecoulement in ecoulements]
    return debits, nuc_max, debut_nuc, demi_nuc


if __name__ == "__main__":
    tic = time.time()
    debits, nuc_max, debut_nuc, demi_nuc = main()
    toc = time.time()
    print(toc-tic)

    plt.plot(np.log10(debits), np.log10(nuc_max))
    plt.xlabel("log10(kg/s)")
    plt.ylabel("log10(Taux de nucleation)")
    plt.figure()
    plt.plot(np.log10(debits), debut_nuc)
    plt.plot(np.log10(debits), demi_nuc)
    plt.xlabel("log10(kg/s)")
    plt.legend(["Position de début de nucléation", "Position de démi-nucléation"])
    plt.figure()
    plt.plot(np.log10(debits), np.array(demi_nuc) - np.array(debut_nuc))
    plt.xlabel("log10(kg/s)")
    plt.ylabel("Distance de démi-nucléation")
    plt.show()
