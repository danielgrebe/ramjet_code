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
H2O = 0.0016
MASSE_MOLAIRE = 2 * 14.5e-3  # kg/mol
GAMMA = (1 - CO2 - H2O) * 1.4 + CO2 * 1.289 + H2O * 1.330
GAMMA = 1.36
GAMMA_CO2 = 1.289
C_p = 1.006e3  # J/(kg*K)

# paramètres
X_STAR_OFFSET = 0.001  # m
N_DEFAULT = 1000

# paramètres particules
PARTICLE_MASS_FLOW_RATE = 10e-3  # kg/s
PART_RADIUS = 2.5e-6
A_N = 4*np.pi*PART_RADIUS**2
BOLTZMAN = 1.380649e-23
CONTACT_PARAMETER = 0.95
AVAGADRO = 6.022e23
MASSE_MOLAIRE_CO2 = 7.3082e-26 * AVAGADRO
DRY_ICE_SURFACE_TENSION = 0.08
RHO_DRY_ICE = 1.6e3  # kg/m^3
MEAN_JUMPING_DISTANCE = 0.4e-9  # m
CO2_VIBRATION_FREQUENCY = 2.9e12
DESORPTION_ENERGY = 2e4  # J/mol
MASSE_PARTICULE = 8.85e-12  # kg

CO2_ETHALPY_SUB = 591e3  # J/kg
C_P_CO2 = 849  #J/kgK


class geometryWarning(Warning):
    def __int__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class Geometry:
    def __init__(self, basename, unit=1e-2, x_star=0.0, nozzle_depth=0.01, friction_on=False):
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
        self.area_ratio = interp1d(self.x_points, self.h_points / self.h_star)
        if friction_on:
            self.coef_friction = self._import_friction(basename)
        else:
            self.coef_friction = lambda x: 0

    def _import_friction(self, basename):
        f_points = np.genfromtxt(basename + ".f.csv").transpose()
        X_f = np.concatenate(([self.x_points[0]], f_points[0] / 100, [self.x_points[-1]]))
        Y_f = np.concatenate(([f_points[1, 0]], f_points[1], [f_points[1, -1]]))
        return interp1d(X_f, Y_f)


    def D_hydraulique(self, x):
        return 4 * self.area(x) / self.perimeter(x)

    def perimeter(self, x):
        return 2 * self.h(x) + 2 * self.nozzle_depth

    def area(self, x):
        return self.h(x) * self.nozzle_depth

    def f(self, x):
        return self.coef_friction(x)


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
        self.masse_molaire = masse_molaire

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


class IdealGasVariable(IdealGas):
    def __init__(self, gamma_inert=1.4, masse_molaire_inert=28.97, c_p_inert=1.006e3, frac_CO2_init=0.0):
        self.gamma_inert = gamma_inert
        self.masse_molaire_inert = masse_molaire_inert
        self.c_p_inert = c_p_inert
        self.frac_CO2_init = 0.0
        super().__init__(self.gamma_var(frac_CO2_init), self.masse_molaire_var(frac_CO2_init), self.c_p_var(frac_CO2_init))

    def gamma_var(self, frac_CO2):
        return frac_CO2 * GAMMA_CO2 + (1 - frac_CO2) * self.gamma_inert

    def masse_molaire_var(self, frac_CO2):
        return frac_CO2 * MASSE_MOLAIRE_CO2 * 1000 + (1 - frac_CO2) * self.masse_molaire_inert

    def c_p_var(self, frac_CO2):
        return frac_CO2 * C_P_CO2 + (1 - frac_CO2) * self.c_p_inert

    def R_var(self, frac_CO2):
        return self.R_UNIVERSAL / self.masse_molaire_var(frac_CO2)

    def pressure(self, temperature, density, frac_CO2=None):
        """
        Pression statique en fonction de la temperature et la densité
        :param temperature: temperature en K
        :param density: masse volumique en kg/m^3
        :return:
        """
        if frac_CO2 is None:
            return super().pressure(temperature, density)
        return density * temperature * self.R_var(frac_CO2)

    def density(self, temperature, pressure, frac_CO2=None):
        """
        Masse volumique selon la loi des gaz parfaits
        :param temperature: temperature en K
        :param pressure: pression en Pa
        :return:
        """
        if frac_CO2 is None:
            return super().density(temperature, pressure)
        return pressure / temperature / self.R_var(frac_CO2)

    def speed_of_sound(self, temperature, frac_CO2=None):
        """

        :param temperature:
        :return:
        """
        if frac_CO2 is None:
            return super().speed_of_sound(temperature)
        return np.sqrt(temperature * self.gamma_var(frac_CO2) * self.R_var(frac_CO2))



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

    def _mach_iso(self, x):
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
        func = lambda mach: 1 / mach * ((1 + (self.gas_prop.gamma - 1) / 2 * mach ** 2) /
                                        ((self.gas_prop.gamma + 1) / 2)) ** ((self.gas_prop.gamma + 1)
                                                                             / (2 * (
                        self.gas_prop.gamma - 1))) - area_ratio
        return fsolve(func, 0.1)

    def _u_iso(self, x):
        return self._mach_iso(x) * self.gas_prop.speed_of_sound(self._temp_iso(x))

    def _rho_iso(self, x):
        return self.gas_prop.density(self._temp_iso(x), self._pressure_iso(x))

    def _temp_iso(self, x):
        return self.init_cond.temperature / self._T_ratio_mach(self._mach_iso(x))

    def _pressure_iso(self, x):
        return self.init_cond.pressure / self._p_ratio_mach(self._mach_iso(x))

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

    temperature = _temp_iso
    rho = _rho_iso
    mach = _mach_iso
    u = _u_iso
    pressure = _pressure_iso


class General_1D_Flow(IsoEcoulement):
    def __init__(self,
                 geo: Geometry,
                 init_cond: InitialConditions,
                 gas_prop: IdealGas,
                 basename=None,
                 n=N_DEFAULT,
                 heat_on=True):
        self.geo = geo
        self.init_cond = init_cond
        self.gas_prop = gas_prop
        self.mach_array = None

        # import heatdata
        if heat_on:
            self.q = self._import_heat(basename)
        else:
            self.q = lambda x: 0
        x_array, mach_array, temp_array, rho_array = self.calculate(n)
        self.mach = interp1d(x_array, mach_array)
        self.temperature = interp1d(x_array, temp_array)
        self.rho = interp1d(x_array, rho_array)
        self.pressure = interp1d(x_array, self.gas_prop.pressure(temp_array, rho_array))
        self.u = interp1d(x_array, self.gas_prop.speed_of_sound(temp_array) * mach_array)

    def calculate(self, n):
        x_array_sub = np.linspace(self.geo.x_points[0], self.geo.x_star + X_STAR_OFFSET, n)
        mach_array_sub = self._mach_iso(x_array_sub)
        temp_array_sub = self._temp_iso(x_array_sub)
        rho_array_sub = self._rho_iso(x_array_sub)
        x_array_super = np.linspace(self.geo.x_star + X_STAR_OFFSET, self.geo.x_points[-1], n)
        mach_array_super = np.zeros(x_array_sub.shape)
        temp_array_super = np.zeros(x_array_sub.shape)
        rho_array_super = np.zeros(x_array_sub.shape)
        mach_array_super[0] = self._mach_iso(x_array_super[0])
        temp_array_super[0] = self._temp_iso(x_array_super[0])
        rho_array_super[0] = self._rho_iso(x_array_super[0])
        for i in range(len(x_array_super) - 1):
            x = x_array_super[i]
            dx = x_array_super[i+1] - x_array_super[i]
            f = self.geo.f(x)
            pressure = self.gas_prop.pressure(temp_array_super[i], rho_array_super[i])
            dq = self.q(x_array_super[i + 1]) - self.q(x_array_super[i])
            dA_A = (self.geo.area(x + dx) -
                    self.geo.area(x)) / self.geo.area(x)
            mach_array_super[i + 1] = np.sqrt(mach_array_super[i] ** 2 +
                                              self._dM2_area(mach_array_super[i],
                                                             dA_A) +
                                              self._dM2_heat(mach_array_super[i],
                                                             dq,
                                                             temp_array_super[i]) +
                                              self._dM2_momentum(mach_array_super[i],
                                                                 f,
                                                                 dx,
                                                                 self.geo.D_hydraulique(x),
                                                                 pressure,
                                                                 self.geo.area(x)) +
                                              self._dM2_mass(mach_array_super[i]) +
                                              self._dM2_mol(mach_array_super[i]) +
                                              self._dM2_gamma(mach_array_super[i]))
            temp_array_super[i + 1] = (temp_array_super[i] +
                                       self._dT_area(temp_array_super[i],
                                                     mach_array_super[i],
                                                     dA_A) +
                                       self._dT_heat(mach_array_super[i],
                                                     dq,
                                                     temp_array_super[i]) +
                                       self._dT_momentum(temp_array_super[i],
                                                         mach_array_super[i],
                                                         f,
                                                         dx,
                                                         self.geo.D_hydraulique(x),
                                                         pressure,
                                                         self.geo.area(x)) +
                                       self._dT_mass(temp_array_super[i],
                                                     mach_array_super[i]) +
                                       self._dT_mol(temp_array_super[i],
                                                    mach_array_super[i]) +
                                       self._dT_gamma(temp_array_super[i]))
            rho_array_super[i + 1] = (rho_array_super[i] +
                                      self._drho_area(rho_array_super[i],
                                                      mach_array_super[i],
                                                      dA_A) +
                                      self._drho_heat(rho_array_super[i],
                                                      mach_array_super[i],
                                                      dq,
                                                      temp_array_super[i]) +
                                      self._drho_momentum(rho_array_super[i],
                                                          mach_array_super[i],
                                                          f,
                                                          dx,
                                                          self.geo.D_hydraulique(x),
                                                          pressure,
                                                          self.geo.area(x)) +
                                      self._drho_mass(rho_array_super[i],
                                                      mach_array_super[i]) +
                                      self._drho_mol(rho_array_super[i],
                                                     mach_array_super[i]) +
                                      self._drho_gamma(rho_array_super[i]))
        x_array = np.concatenate((x_array_sub, x_array_super))
        mach_array = np.concatenate((mach_array_sub, mach_array_super))
        temp_array = np.concatenate((temp_array_sub, temp_array_super))
        rho_array = np.concatenate((rho_array_sub, rho_array_super))
        return x_array, mach_array, temp_array, rho_array

    def _import_heat(self, basename):
        q_points = np.genfromtxt(basename + ".q.csv").transpose()
        X_q = np.concatenate(([self.geo.x_points[0]], q_points[0] / 100, [self.geo.x_points[-1]]))
        Y_q = np.concatenate((np.array([0]), q_points[1], [q_points[1, -1]])) * 1000
        # Y_q = np.zeros(Y_q.shape)
        return interp1d(X_q, Y_q)

    def _dM2_area(self, M, dA_A, gamma=None):
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = - 2 * (1 + (gamma - 1) / 2 * M ** 2) / (1 - M ** 2)
        return M ** 2 * coef * dA_A

    def _dM2_heat(self, M, dq, T, c_p=None, gamma=None, dWx=0, dH=0):
        """

        :param M: nombre de M
        :param dq: Apport de chaleur (J)
        :param T: Température (K)
        :param c_p: chaleur spécifique à pression constant (J/kgK)
        :param gamma: ratio de chaleur spécifiques
        :param dWx: Travail (J)
        :param dH: variation d'énergie (J)
        :return: Variation du nombre de M au carré
        """
        if c_p is None:
            c_p = self.gas_prop.c_p
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = (1 + gamma * M ** 2) / (1 - M ** 2)
        variable = (dq - dWx + dH) / (c_p * T)
        return M ** 2 * coef * variable

    def _dM2_momentum(self, M, f, dx, D, p, A, dX=0, dw_w=0, y=1, gamma=None):
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = gamma * M ** 2 * (1 + (gamma - 1) / 2 * M ** 2) / (1 - M ** 2)
        variable = 4 * f * dx / D + 2 * dX / (gamma * p * A * M ** 2) - 2 * y * dw_w
        return M ** 2 * coef * variable

    def _dM2_mass(self, M, dw_w=0, gamma=None):
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = 2 * (1 + gamma * M ** 2) * (1 + (gamma - 1) / 2 * M ** 2) / (1 - M ** 2)
        variable = dw_w
        return M ** 2 * coef * variable

    def _dM2_mol(self, M, dW_W=0, gamma=None):
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = - (1 + gamma * M ** 2) / (1 - M ** 2)
        variable = dW_W
        return M ** 2 * coef * variable

    def _dM2_gamma(self, M, dgamma_gamma=0, gamma=None):
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = -1
        variable = dgamma_gamma
        return M ** 2 * coef * variable

    def _dT_area(self, T, M, dA_A, gamma=None):
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = (gamma - 1) * M ** 2 / (1 - M ** 2)
        return T * coef * dA_A

    def _dT_heat(self, M, dq, T, c_p=None, gamma=None, dWx=0, dH=0):
        """

        :param M: nombre de M
        :param dq: Apport de chaleur (J)
        :param T: Température (K)
        :param c_p: chaleur spécifique à pression constant (J/kgK)
        :param gamma: ratio de chaleur spécifiques
        :param dWx: Travail (J)
        :param dH: variation d'énergie (J)
        :return: Variation du nombre de M au carré
        """
        if c_p is None:
            c_p = self.gas_prop.c_p
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = (1 - gamma * M ** 2) / (1 - M ** 2)
        variable = (dq - dWx + dH) / (c_p * T)
        return T * coef * variable

    def _dT_momentum(self, T, M, f, dx, D, p, A, dX=0, dw_w=0, y=1, gamma=None):
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = - gamma * (gamma - 1) * M ** 4 / (2 * (1 - M ** 2))
        variable = 4 * f * dx / D + 2 * dX / (gamma * p * A * M ** 2) - 2 * y * dw_w
        return T * coef * variable

    def _dT_mass(self, T, M, dw_w=0, gamma=None):
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = - (gamma - 1) * M ** 2 * (1 + gamma * M ** 2) / (1 - M ** 2)
        variable = dw_w
        return T * coef * variable

    def _dT_gamma(self, T, dgamma_gamma=0, gamma=None):
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = 0
        variable = dgamma_gamma
        return T * coef * variable

    def _dT_mol(self, T, M, dW_W=0, gamma=None):
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = (gamma - 1) * M ** 2 / (1 - M ** 2)
        variable = dW_W
        return T * coef * variable

    def _drho_area(self, rho, M, dA_A, gamma=None):
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = M ** 2 / (1 - M ** 2)
        return rho * coef * dA_A

    def _drho_heat(self, rho, M, dq, T, c_p=None, gamma=None, dWx=0, dH=0):
        """

        :param M: nombre de M
        :param dq: Apport de chaleur (J)
        :param T: Température (K)
        :param c_p: chaleur spécifique à pression constant (J/kgK)
        :param gamma: ratio de chaleur spécifiques
        :param dWx: Travail (J)
        :param dH: variation d'énergie (J)
        :return: Variation du nombre de M au carré
        """
        if c_p is None:
            c_p = self.gas_prop.c_p
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = - 1 / (1 - M ** 2)
        variable = (dq - dWx + dH) / (c_p * T)
        return rho * coef * variable

    def _drho_momentum(self, rho, M, f, dx, D, p, A, dX=0, dw_w=0, y=1, gamma=None):
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = - gamma * M ** 2 / (2 * (1 - M ** 2))
        variable = 4 * f * dx / D + 2 * dX / (gamma * p * A * M ** 2) - 2 * y * dw_w
        return rho * coef * variable

    def _drho_mass(self, rho, M, dw_w=0, gamma=None):
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = - (gamma + 1) * M ** 2 / (1 - M ** 2)
        variable = dw_w
        return rho * coef * variable

    def _drho_mol(self, rho, M, dW_W=0, gamma=None):
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = 1 / (1 - M ** 2)
        variable = dW_W
        return rho * coef * variable

    def _drho_gamma(self, rho, dgamma_gamma=0, gamma=None):
        if gamma is None:
            gamma = self.gas_prop.gamma
        coef = 0
        variable = dgamma_gamma
        return rho * coef * variable


class Condensing_Ecoulement(General_1D_Flow):
    def __init__(self,
                 geo: Geometry,
                 init_cond: InitialConditions,
                 gas_prop: IdealGasVariable,
                 basename,
                 part_mass_flow_rate,
                 n=N_DEFAULT):
        self.part_mass_flow_rate = part_mass_flow_rate
        self.geo = geo
        self.init_cond = init_cond
        self.gas_prop = gas_prop
        self.mach_array = None

        # import heatdata
        self.q = lambda x: 0
        x_array, mach_array, temp_array, rho_array, frac_co2_array, nuc_array = self.calculate(n)
        self.mach = interp1d(x_array, mach_array)
        self.temperature = interp1d(x_array, temp_array)
        self.rho = interp1d(x_array, rho_array)
        self.pressure = interp1d(x_array, self.gas_prop.pressure(temp_array, rho_array))
        self.u = interp1d(x_array, self.gas_prop.speed_of_sound(temp_array) * mach_array)
        self.nucleation_rate = interp1d(x_array, nuc_array)
        self.frac_co2 = interp1d(x_array, frac_co2_array)

    def calculate(self, n):
        # Initialize subsonic arrays with isentropic flow
        x_array_sub = np.linspace(self.geo.x_points[0], self.geo.x_star + X_STAR_OFFSET, n)
        mach_array_sub = self._mach_iso(x_array_sub)
        temp_array_sub = self._temp_iso(x_array_sub)
        rho_array_sub = self._rho_iso(x_array_sub)
        nuc_array_sub = np.zeros(x_array_sub.shape)
        frac_co2_array_sub = np.ones(x_array_sub.shape) * CO2

        # Initialize supersonic arrays
        x_array_super = np.linspace(self.geo.x_star + X_STAR_OFFSET, self.geo.x_points[-1], n)
        mach_array_super = np.zeros(x_array_sub.shape)
        temp_array_super = np.zeros(x_array_sub.shape)
        rho_array_super = np.zeros(x_array_sub.shape)
        nuc_array_super = np.zeros(x_array_super.shape)
        q_array_super = np.zeros(x_array_super.shape)
        gamma_array_super = np.zeros(x_array_super.shape)
        masse_molaire_array_super = np.zeros(x_array_super.shape)
        frac_co2_array_super = np.zeros(x_array_super.shape)
        m_dot_array_super = np.zeros(x_array_super.shape)
        nuc_array_super[0] = 0
        frac_co2_array_super[0] = CO2
        mach_array_super[0] = self._mach_iso(x_array_super[0])
        temp_array_super[0] = self._temp_iso(x_array_super[0])
        rho_array_super[0] = self._rho_iso(x_array_super[0])
        m_dot_array_super[0] = (self._rho_iso(x_array_super[0]) *
                                self._u_iso(x_array_super[0]) *
                                self.geo.area(x_array_super[0]))

        # Integrate supersonic flow
        for i in range(len(x_array_super) - 1):
            # convenience variables
            frac_co2 = frac_co2_array_super[i]
            x = x_array_super[i]
            dx = x_array_super[i+1] - x_array_super[i]
            f = self.geo.f(x)
            pressure = self.gas_prop.pressure(temp_array_super[i], rho_array_super[i], frac_co2)
            A = self.geo.area(x)
            dA_A = (self.geo.area(x + dx) -
                    self.geo.area(x)) / self.geo.area(x)
            T = temp_array_super[i]
            M = mach_array_super[i]
            u = M * self.gas_prop.speed_of_sound(T, frac_co2)
            mdot = m_dot_array_super[i]
            gamma = gamma_array_super[i] = self.gas_prop.gamma_var(frac_co2)
            masse_molaire = masse_molaire_array_super[i] = self.gas_prop.masse_molaire_var(frac_co2)
            c_p = self.gas_prop.c_p_var(frac_co2)

            # calcul de nucleation
            J = nuc_array_super[i] = self._nucleation_rate(frac_co2_array_super[i], rho_array_super[i], temp_array_super[i])
            dmdot = - J * MASSE_MOLAIRE_CO2 / AVAGADRO * self.part_mass_flow_rate * dx / (MASSE_PARTICULE * u)
            dmdot_mdot = dmdot / mdot
            m_dot_array_super[i + 1] = mdot + dmdot
            dfrac_CO2 = dmdot_mdot * (1 - frac_co2_array_super[i])
            frac_co2_array_super[i + 1] = frac_co2_array_super[i] + dfrac_CO2
            dq = -dmdot_mdot * CO2_ETHALPY_SUB # / u / A
            q_array_super[i] = dq
            dgamma_gamma = (self.gas_prop.gamma_var(frac_co2 + dfrac_CO2) - gamma)/gamma
            dmmol_mmol = (self.gas_prop.masse_molaire_var(frac_co2 + dfrac_CO2) - masse_molaire)/masse_molaire


            # iterate variables
            mach_array_super[i + 1] = np.sqrt(mach_array_super[i] ** 2 +
                                              self._dM2_area(mach_array_super[i],
                                                             dA_A,
                                                             gamma) +
                                              self._dM2_heat(mach_array_super[i],
                                                             dq,
                                                             temp_array_super[i],
                                                             c_p=c_p,
                                                             gamma=gamma) +
                                              self._dM2_momentum(mach_array_super[i],
                                                                 f,
                                                                 dx,
                                                                 self.geo.D_hydraulique(x),
                                                                 pressure,
                                                                 self.geo.area(x),
                                                                 dw_w=dmdot_mdot,
                                                                 gamma=gamma) +
                                              self._dM2_mass(mach_array_super[i],
                                                             dw_w=dmdot_mdot,
                                                             gamma=gamma) +
                                              self._dM2_mol(mach_array_super[i],
                                                            dW_W=dmmol_mmol,
                                                            gamma=gamma) +
                                              self._dM2_gamma(mach_array_super[i],
                                                              dgamma_gamma=dgamma_gamma,
                                                              gamma=gamma))
            temp_array_super[i + 1] = (temp_array_super[i] +
                                       self._dT_area(temp_array_super[i],
                                                     mach_array_super[i],
                                                     dA_A,
                                                     gamma=gamma) +
                                       self._dT_heat(mach_array_super[i],
                                                     dq,
                                                     temp_array_super[i],
                                                     gamma=gamma) +
                                       self._dT_momentum(temp_array_super[i],
                                                         mach_array_super[i],
                                                         f,
                                                         dx,
                                                         self.geo.D_hydraulique(x),
                                                         pressure,
                                                         self.geo.area(x),
                                                         dw_w=dmdot_mdot,
                                                         gamma=gamma) +
                                       self._dT_mass(temp_array_super[i],
                                                     mach_array_super[i],
                                                     dw_w=dmdot_mdot,
                                                     gamma=gamma) +
                                       self._dT_mol(temp_array_super[i],
                                                    mach_array_super[i],
                                                    dW_W=dmmol_mmol,
                                                    gamma=gamma) +
                                       self._dT_gamma(temp_array_super[i],
                                                      dgamma_gamma=dgamma_gamma,
                                                      gamma=gamma))
            rho_array_super[i + 1] = (rho_array_super[i] +
                                      self._drho_area(rho_array_super[i],
                                                      mach_array_super[i],
                                                      dA_A,
                                                      gamma=gamma) +
                                      self._drho_heat(rho_array_super[i],
                                                      mach_array_super[i],
                                                      dq,
                                                      temp_array_super[i],
                                                      gamma=gamma) +
                                      self._drho_momentum(rho_array_super[i],
                                                          mach_array_super[i],
                                                          f,
                                                          dx,
                                                          self.geo.D_hydraulique(x),
                                                          pressure,
                                                          self.geo.area(x),
                                                          dw_w=dmdot_mdot,
                                                          gamma=gamma) +
                                      self._drho_mass(rho_array_super[i],
                                                      mach_array_super[i],
                                                      dw_w=dmdot_mdot,
                                                      gamma=gamma) +
                                      self._drho_mol(rho_array_super[i],
                                                     mach_array_super[i],
                                                     dW_W=dmmol_mmol,
                                                     gamma=gamma) +
                                      self._drho_gamma(rho_array_super[i],
                                                       dgamma_gamma=dgamma_gamma,
                                                       gamma=gamma))
        x_array = np.concatenate((x_array_sub, x_array_super))
        mach_array = np.concatenate((mach_array_sub, mach_array_super))
        temp_array = np.concatenate((temp_array_sub, temp_array_super))
        rho_array = np.concatenate((rho_array_sub, rho_array_super))
        frac_co2_array = np.concatenate((frac_co2_array_sub, frac_co2_array_super))
        nuc_array = np.concatenate((nuc_array_sub, nuc_array_super))
        return x_array, mach_array, temp_array, rho_array, frac_co2_array, nuc_array

    def _nucleation_rate(self, frac_co2, rho, temp):
        a_n = A_N
        m = CONTACT_PARAMETER
        m_co2 = MASSE_MOLAIRE_CO2 / AVAGADRO
        sigma = DRY_ICE_SURFACE_TENSION
        t_part = temp
        T_n = temp
        rho_co2 = RHO_DRY_ICE
        p_co2 = self.gas_prop.pressure(temp, rho, frac_co2) * frac_co2
        # p_sat_co2 = 100000*10**(6.81228-1301.779/(T_n-3.494))
        p_sat_co2 = 1.38e12 * np.exp(-3182.48/t_part)
        S = p_co2 / p_sat_co2
        if S < 1.001:
            S = 1.001
        k = BOLTZMAN
        r_crit = 2 * m_co2 * sigma / rho_co2 / k / T_n / np.log(S)
        x = PART_RADIUS / r_crit
        d = MEAN_JUMPING_DISTANCE
        nu = CO2_VIBRATION_FREQUENCY
        theta = 0.1 #TODO: investiguer d'ou vient ce valeur
        delta_f_sd = DESORPTION_ENERGY / 10 / AVAGADRO
        n_crit = 4 * np.pi * r_crit ** 3 * rho_co2 * 1000 / 3 / m_co2
        delta_f_hom = 4* np.pi * sigma * r_crit ** 2 / 3
        c_1s = p_co2 / nu / np.sqrt(2*np.pi * m_co2 * k * T_n)*np.exp(DESORPTION_ENERGY/AVAGADRO/k/T_n)
        z_hom = np.sqrt(delta_f_hom/3/np.pi/k/T_n/n_crit**2)
        z_het = z_hom * np.sqrt(4 / (2 + ((1 - m*x)*(2 - 4 * m * x - (m ** 2 - 3)*x**2))/(1-2*m*x+x**2)**1.5))
        beta_het = np.pi * 2 * r_crit * np.sin(theta) * d * c_1s * nu * np.exp(-delta_f_sd/k/T_n)
        delta_f_het = self._f(m, x) * delta_f_hom
        return a_n * z_het * beta_het * c_1s * np.exp(-delta_f_het / k / T_n)

    def _f(self, m, x):
        phi = np.sqrt(1 - 2*m*x + x ** 2)
        k = (x-m)/ phi
        return 0.5 * (1 + (1 - m * x) ** 3 / phi ** 3 + x ** 3 * (2 - 3 * k + k ** 3) + 3 * m * x ** 2 * (k - 1))

    # def _dfrac_co2(self, nucleation_rate, rho_gaz, drho_gaz, frac_co2, dx, u, area):
    #     """

    #     :param nucleation_rate: Taux de nucléation en molécules par noyau par seconde
    #     :param rho_gaz: Masse volumique du gaz
    #     :param drho_gaz: Variation de la masse volumique du gaz
    #     :param frac_co2: Fraction molaire du CO2
    #     :param dx: Variation de la position
    #     :param u: Vitesse de l'écoulement
    #     :param area: Section de l'écoulement
    #     :return: variation de la fraction molaire de CO2
    #     """
    #     n_tot = rho_gaz / self.gas_prop.masse_molaire * 1000
    #     dn_tot = drho_gaz / self.gas_prop.masse_molaire * 1000  # valide car on néglige le variation de la masse molaire du gaz
    #     n_particles = self.part_mass_flow_rate / u / area / MASSE_PARTICULE
    #     dn_co2 = -nucleation_rate * n_particles * dx / u / AVAGADRO + drho_gaz / rho_gaz * n_tot * frac_co2
    #     n_co2 = n_tot * frac_co2
    #     return (dn_co2 * n_tot - dn_tot * n_co2) / n_tot ** 2

def main():
    geo = Geometry(DATA_BASENAME)
    init_cond = InitialConditions(P_0, T_0)
    gas_prop = IdealGasVariable(frac_CO2_init=CO2)
    iso = IsoEcoulement(geo, init_cond, gas_prop)
    heat = General_1D_Flow(geo, init_cond, gas_prop, DATA_BASENAME)
    cond = Condensing_Ecoulement(geo, init_cond, gas_prop, DATA_BASENAME, PARTICLE_MASS_FLOW_RATE, n=int(3e3))

    # -----------------------------------------------------------------------------
    #                               PLOTTING
    # -----------------------------------------------------------------------------
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    X_iso = np.linspace(geo.x_points[0], geo.x_points[-1], 100)
    ax0.plot(X_iso, iso.pressure(X_iso))
    ax0.plot(X_iso, cond.pressure(X_iso))
    ax0.set_xlabel('x (m)')
    ax0.set_ylabel('P (Pa)')
    ax1.plot(X_iso, iso.temperature(X_iso))
    ax1.plot(X_iso, cond.temperature(X_iso))
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('T (K)')
    ax2.plot(X_iso, iso.mach(X_iso))
    ax2.plot(X_iso, cond.mach(X_iso))
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('Mach')
    ax3.plot(X_iso, iso.u(X_iso) / iso.mach(X_iso))
    ax3.plot(X_iso, cond.u(X_iso) / cond.mach(X_iso))
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('u (m/s)')
    fig.legend(["Isentropique", "Condensation"])

    f = plt.figure()
    x_tanimura = np.linspace(0, 0.105, 100)
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    im = plt.imread(DATA_BASENAME + ".PT.png")
    implot = ax.imshow(im, origin="upper", extent=(0, 12, 100, 240), aspect='auto')
    ax.plot(x_tanimura * 100, cond.temperature(x_tanimura))
    ax.plot(x_tanimura * 100, iso.temperature(x_tanimura))
    ax.set_xlim([0, 12])
    ax.set_ylim([100, 240])

    f2 = plt.figure()
    plt.plot(X_iso, cond.frac_co2(X_iso))
    plt.xlabel("Position (m)")
    plt.ylabel("Fraction massique de CO2")
    f3 = plt.figure()
    plt.plot(X_iso, cond.nucleation_rate(X_iso))
    plt.xlabel("Position (m)")
    plt.ylabel("Taux de nucléation")
    plt.show()


if __name__ == "__main__":
    main()
