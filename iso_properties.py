import numpy as np
from cnst import *


def area_ratio_Ma(Ma):
	"""Ratio A/A* en fonction du nombre de Mach"""
	return 1 / Ma * ((1 + (GAMMA - 1) / 2 * Ma ** 2 ) /
		( (GAMMA + 1) / 2 )) ** ((GAMMA + 1) / (2 * (GAMMA - 1)))


def T_ratio_Ma(Ma):
	"""Ratio T_0/T en fonction du nombre de Mach

	Hypothèses:
	 - gaz parfait
	"""
	return 1 + (GAMMA - 1) / 2 * Ma ** 2


def p_ratio_Ma(Ma):
	"""Ratio p_0/p en fonction du nombre de Mach

	Hypothèses:
	 - isentropique
	 - gaz parfait
	"""
	return T_ratio_Ma(Ma) ** (GAMMA / (GAMMA - 1))


def rho_ratio_Ma(Ma):
	"""Ratio rho_0/rho en fonction du nombre de Mach

	Hypothèses:
	 - isentropique
	 - gaz parfait
	"""
	return T_ratio_Ma(Ma) ** (1 / (GAMMA - 1))


def rho_ratio_T(T, T0):
	"""Ratio rho_0/rho en fonction du nombre de Mach

	Hypothèses:
	 - isentropique
	 - gaz parfait
	"""
	return (T0/T) ** (1 / (GAMMA - 1))


def speed_of_sound_Ma(Ma):
	"""Vitesse du son en fonction du nombre de mach

	Hypothèses:
	 - gaz parfait
	"""
	return np.sqrt(GAMMA * R * T_0 / T_ratio_Ma(Ma))


def speed_of_sound_T(T):
	"""Vitesse du son en fonction de la temperature (K)

	Hypothèses:
	 - gaz parfait
	"""
	return np.sqrt(GAMMA * R * T)


def x_Ma(Ma):
	"""Position en fonction du nombre de Mach
	"""
	return x_A_Astar(area_ratio_Ma(Ma), Ma)