import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
from collections.abc import Iterable

DATA_BASENAME = "data/tanimura_2015_run4"

# Conditions initials
P_0 = 2.026e5  # Pa
T_0 = 288.2  # K
# Propriétés du gaz
CO2 = 0.212  # fraction massique du CO_2
MASSE_MOLAIRE = 2*14.5e-3  # kg/mol
GAMMA = (1-CO2)*1.4 + CO2*1.289
C_p = 1.006e3  # J/(kg*K)


class geometryWarning(Warning):
    def __int__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class Geometry:
    def __init__(self, basename, unit=1e-2, x_star=0.0, nozzle_depth=0.01):
        """
        Initialize geometry
        :param basename: Basename of data file
        :param unit: conversion of data file unit to metres (default=cm)
        """
        points = np.genfromtxt(basename + ".geo.csv", skip_header=1).transpose()
        self.x_points = points[0] * unit
        self.h_points = points[1] * unit
        self.x_star = x_star
        star_pos = np.where(self.x_points == self.x_star)
        self.h = interp1d(self.x_points, self.h_points)
        self.nozzle_depth = nozzle_depth
        if len(star_pos) == 1:
            self.h_star = float(self.h_points[int(star_pos[0])])
        else:
            warnings.warn("Le col n'est pas spécifié dans les données. Le col sera déterminé par interpolation",
                          geometryWarning)
            self.h_star = self.h(self.x_star)
        self.area = interp1d(self.x_points, self.h_points * self.nozzle_depth)
        self.area_ratio = interp1d(self.x_points, self.h_points / self.h_star)


class InitialConditions:
    def __init__(self, p0, temp0):
        self.pressure = p0
        self.temperature = temp0


class GasProperties:
    pass


class IdealGas(GasProperties):
    R_UNIVERSAL = 8314  # J/(K*kmol)

    def __init__(self, gamma=1.4, masse_molaire=28.97, c_p=1.006e3):
        self.gamma = gamma
        self.R = self.R_UNIVERSAL / masse_molaire
        self.c_p = c_p

    def pressure(self, temperature, density):
        """
        Pression statique en fonction de la temperature et la densité
        :param temperature: temperature en K
        :param density: masse volumique en kg/m^3
        :return:
        """
        return density * temperature * self.R

    def density(self, temperature, pressure):
        """
        Masse volumique selon la loi des gaz parfaits
        :param temperature: temperature en K
        :param pressure: pression en Pa
        :return:
        """
        return pressure / temperature / self.R

    def speed_of_sound(self, temperature):
        """

        :param temperature:
        :return:
        """
        return np.sqrt(temperature * self.gamma * self.R)


class Ecoulement:
    def __init__(self,
                 geo: Geometry,
                 init_cond: InitialConditions,
                 gas_prop: GasProperties):
        self.geo = geo
        self.init_cond = init_cond
        self.gas_prop = gas_prop


class IsoEcoulement(Ecoulement):
    def __init__(self,
                 geo: Geometry,
                 init_cond: InitialConditions,
                 gas_prop: IdealGas,
                 initialize_mach=False):
        self.geo = geo
        self.init_cond = init_cond
        self.gas_prop = gas_prop
        if initialize_mach:
            self.mach_array = self._initialize_mach()
        else:
            self.mach_array = None

    def mach(self, x):
        single_output = False
        if not isinstance(x, Iterable):
            x = [x]
            single_output = True
        if self.mach_array is None:
            mach = np.zeros(len(x))
            for i, x_value in enumerate(x):
                if x_value < self.geo.x_star:
                    mach[i] = fsolve(lambda m: self._area_ratio_mach(m) - self.geo.area_ratio(x_value), 0.1)
                elif x_value > self.geo.x_star:
                    mach[i] = fsolve(lambda m: self._area_ratio_mach(m) - self.geo.area_ratio(x_value), 2)
                else:
                    mach[i] = 1.0
            if single_output:
                return mach[0]
            else:
                return mach

    def _area_ratio_mach(self, mach):
        """Ratio A/A* en fonction du nombre de Mach"""
        return 1 / mach * ((1 + (self.gas_prop.gamma - 1) / 2 * mach ** 2) /
                           ((self.gas_prop.gamma + 1) / 2)) ** ((self.gas_prop.gamma + 1)
                                                                / (2 * (self.gas_prop.gamma - 1)))

    def _mach_area_ratio_subsonic(self, area_ratio):
        """Ratio A/A* en fonction du nombre de Mach"""
        func = lambda mach :1 / mach * ((1 + (self.gas_prop.gamma - 1) / 2 * mach ** 2) /
                           ((self.gas_prop.gamma + 1) / 2)) ** ((self.gas_prop.gamma + 1)
                                                                / (2 * (self.gas_prop.gamma - 1))) - area_ratio
        return fsolve(func, 0.1)

    def u(self, x):
        return self.mach(x) * self.gas_prop.speed_of_sound(self.temperature(x))

    def rho(self, x):
        return self.gas_prop.density(self.temperature(x), self.pressure(x))

    def temperature(self, x):
        return self.init_cond.temperature / self._T_ratio_mach(self.mach(x))

    def pressure(self, x):
        return self.init_cond.pressure / self._p_ratio_mach(self.mach(x))

    def _initialize_mach(self):
        raise NotImplementedError("Precalculated mach array not yet implemented")

    def _T_ratio_mach(self, m):
        return 1 + (self.gas_prop.gamma - 1) / 2 * m ** 2

    def _p_ratio_mach(self, m):
        """Ratio p_0/p en fonction du nombre de Mach

        Hypothèses:
         - isentropique
         - gaz parfait
        """
        return self._T_ratio_mach(m) ** (self.gas_prop.gamma / (self.gas_prop.gamma - 1))


class HeatEcoulement(IsoEcoulement):
    def __init__(self,
                 geo: Geometry,
                 init_cond: InitialConditions,
                 gas_prop: IdealGas,
                 basename):
        self.geo = geo
        self.init_cond = init_cond
        self.gas_prop = gas_prop

        # import heatdata
        self.q = self._import_heat(basename)
        x_array, mach_array, temp_array, rho_array = self.calculate()
        self.mach = interp1d(x_array, mach_array)
        self.temperature = interp1d(x_array, temp_array)
        self.rho = interp1d(x_array, rho_array)
        self.pressure = interp1d(x_array, self.gas_prop.pressure(temp_array, rho_array))
        self.u = interp1d(x_array, self.gas_prop.speed_of_sound(temp_array) * mach_array)

    def calculate(self, n=1000):
        x_array_sub = np.linspace(self.geo.x_points[0], self.geo.x_star, n)
        mach_array_sub = np.zeros(x_array_sub.shape)
        temp_array_sub = np.zeros(x_array_sub.shape)
        rho_array_sub = np.zeros(x_array_sub.shape)
        mach_array_sub[0] = self._mach_area_ratio_subsonic(self.geo.area_ratio(x_array_sub[0]))
        temp_array_sub[0] = self.init_cond.temperature / self._T_ratio_mach(mach_array_sub[0])
        rho_array_sub[0] = self.gas_prop.density(temp_array_sub[0], self.init_cond.pressure / self._p_ratio_mach(mach_array_sub[0]))
        for i in range(len(x_array_sub) - 1):
            dq = self.q(x_array_sub[i + 1]) - self.q(x_array_sub[i])
            mach_array_sub[i + 1] = np.sqrt(mach_array_sub[i] ** 2 +
                                        self._dM2_heat(mach_array_sub[i],
                                                       (self.geo.area(x_array_sub[i + 1]) - self.geo.area(
                                                           x_array_sub[i])) / self.geo.area(x_array_sub[i]),
                                                       dq,
                                                       temp_array_sub[i]))
            temp_array_sub[i + 1] = temp_array_sub[i] + self._dT_heat(mach_array_sub[i],
                                                              (self.geo.area(x_array_sub[i + 1]) - self.geo.area(
                                                                  x_array_sub[i])) / self.geo.area(x_array_sub[i]),
                                                              dq,
                                                              temp_array_sub[i])
            rho_array_sub[i + 1] = rho_array_sub[i] + self._drho_heat(mach_array_sub[i],
                                                              (self.geo.area(x_array_sub[i + 1]) - self.geo.area(
                                                                  x_array_sub[i])) / self.geo.area(x_array_sub[i]),
                                                              dq,
                                                              temp_array_sub[i],
                                                              rho_array_sub[i])
        x_array_super = np.linspace(self.geo.x_star, self.geo.x_points[-1], n)
        mach_array_super = np.zeros(x_array_sub.shape)
        temp_array_super = np.zeros(x_array_sub.shape)
        rho_array_super = np.zeros(x_array_sub.shape)
        mach_array_super[0] = 1.005
        temp_array_super[0] = temp_array_sub[-1]
        rho_array_super[0] = rho_array_sub[-1]
        for i in range(len(x_array_super) - 1):
            dq = self.q(x_array_super[i + 1]) - self.q(x_array_super[i])
            mach_array_super[i + 1] = np.sqrt(mach_array_super[i] ** 2 +
                                            self._dM2_heat(mach_array_super[i],
                                                           (self.geo.area(x_array_super[i + 1]) - self.geo.area(
                                                               x_array_super[i])) / self.geo.area(x_array_super[i]),
                                                           dq,
                                                           temp_array_super[i]))
            temp_array_super[i + 1] = temp_array_super[i] + self._dT_heat(mach_array_super[i],
                                                                      (self.geo.area(x_array_super[i + 1]) - self.geo.area(
                                                                          x_array_super[i])) / self.geo.area(x_array_super[i]),
                                                                      dq,
                                                                      temp_array_super[i])
            rho_array_super[i + 1] = rho_array_super[i] + self._drho_heat(mach_array_super[i],
                                                                      (self.geo.area(x_array_super[i + 1]) - self.geo.area(
                                                                          x_array_super[i])) / self.geo.area(x_array_super[i]),
                                                                      dq,
                                                                      temp_array_super[i],
                                                                      rho_array_super[i])

        x_array = np.concatenate((x_array_sub, x_array_super))
        mach_array = np.concatenate((mach_array_sub, mach_array_super))
        temp_array = np.concatenate((temp_array_sub, temp_array_super))
        rho_array = np.concatenate((rho_array_sub, rho_array_super))
        return x_array, mach_array, temp_array, rho_array

    def _import_heat(self, basename):
        q_points = np.genfromtxt(basename + ".q.csv").transpose()
        X_q = np.concatenate(([self.geo.x_points[0]], q_points[0] / 100, [self.geo.x_points[-1]]))
        Y_q = np.concatenate((np.array([0]), q_points[1], [q_points[1, -1]])) * 1000
        return interp1d(X_q, Y_q)

    def _dM2_heat(self, mach, dA_A, dq, T):
        return mach ** 2 * (dA_A * -2 * (1 + (self.gas_prop.gamma - 1) / 2 * mach ** 2) / (1 - mach ** 2)
                            + dq / self.gas_prop.c_p / T * (1 + self.gas_prop.gamma * mach ** 2) / (1 - mach ** 2))

    def _dT_heat(self, mach, dA_A, dq, T):
        return T * (dA_A * (self.gas_prop.gamma - 1) * mach ** 2 / (1 - mach ** 2) +
                    dq / self.gas_prop.c_p / T * (1 - self.gas_prop.gamma * mach ** 2) / (1 - mach ** 2))

    def _drho_heat(self, mach, dA_A, dq, T, rho):
        return rho * (dA_A * mach ** 2 / (1 - mach ** 2) - dq / self.gas_prop.c_p / T / (1 - mach ** 2))

def main():
    geo = Geometry(DATA_BASENAME)
    init_cond = InitialConditions(P_0, T_0)
    gas_prop = IdealGas(gamma=GAMMA, masse_molaire=MASSE_MOLAIRE * 1000, c_p=C_p)
    iso = IsoEcoulement(geo, init_cond, gas_prop)
    heat = HeatEcoulement(geo, init_cond, gas_prop, DATA_BASENAME)

    # -----------------------------------------------------------------------------
    #                               PLOTTING
    # -----------------------------------------------------------------------------
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    X_iso = np.linspace(geo.x_points[0], geo.x_points[-1], 100)
    ax0.plot(X_iso, iso.pressure(X_iso))
    ax0.plot(X_iso, heat.pressure(X_iso))
    ax0.set_xlabel('x (m)')
    ax0.set_ylabel('P (Pa)')
    ax1.plot(X_iso, iso.temperature(X_iso))
    ax1.plot(X_iso, heat.temperature(X_iso))
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('T (K)')
    ax2.plot(X_iso, iso.mach(X_iso))
    ax2.plot(X_iso, heat.mach(X_iso))
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('Mach')
    ax3.plot(X_iso, iso.u(X_iso) / iso.mach(X_iso))
    ax3.plot(X_iso, heat.u(X_iso) / heat.mach(X_iso))
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('u (m/s)')

    f = plt.figure()
    x_tanimura = np.linspace(0, 0.105, 100)
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    im = plt.imread(DATA_BASENAME + ".PT.png")
    implot = ax.imshow(im, origin="upper", extent=(0, 12, 100, 240), aspect='auto')
    ax.plot(x_tanimura * 100, heat.temperature(x_tanimura))
    ax.plot(x_tanimura * 100, iso.temperature(x_tanimura))
    ax.set_xlim([0, 12])
    ax.set_ylim([100, 240])
    plt.show()


if __name__ == "__main__":
    main()
