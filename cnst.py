import numpy as np
from scipy.interpolate import interp1d

DATA_BASENAME = "data/tanimura_2015_run4"

# Propriétés du gaz
CO2 = 0.212  # fraction massique du CO_2
MASSE_MOLAIRE = 2*14.5e-3  # kg/mol
IDEAL_GAS = 8.314  # J/(K*mol)
GAMMA = (1-CO2)*1.4 + CO2*1.289
C_p = 1.006e3  # J/(kg*K)
C_v = C_p / GAMMA  # J/(kg*K)
R = C_p -C_v
H_s = CO2 * 591e3  # J/kg enthalpy of sublimation du mélange

# Conditions initials
P_0 = 2.026e5  # Pa
T_0 = 288.2  # K
u_0 = 0  # m/s
rho_0 = P_0 / (R * T_0)


points = np.genfromtxt(DATA_BASENAME + ".geo.csv", skip_header=1).transpose()
X_points = points[0]/100
H_points = points[1]/100
X_star = 0.0
star_pos = int(np.where(X_points==X_star)[0])
H_star = float(H_points[star_pos])
NOZZLE_DEPTH = 0.01  # m
A_star = H_star * NOZZLE_DEPTH
A_Astar = interp1d(X_points, H_points/H_star)  # ratio
x_pre_col = interp1d(H_points[:star_pos + 1]/H_star, X_points[:star_pos + 1])
x_post_col = interp1d(H_points[star_pos:]/H_star, X_points[star_pos:])

# Paramètres condensation
COND_LEN = 0.05  # m: longueur sur lequel tout le CO2 condense
dqdx = H_s / COND_LEN  # J/(kg*m)

# Données de chaleur latente
q_points = np.genfromtxt(DATA_BASENAME + ".q.csv").transpose()
X_q = np.concatenate(([X_points[0]], q_points[0]/100, [X_points[-1]]))
Y_q = np.concatenate((np.array([0]), q_points[1], [q_points[1,-1]]))*1000
q_func = interp1d(X_q, Y_q)


def x_A_Astar(A_Astar, Ma):
	if Ma < 1:
		return x_pre_col(A_Astar)
	elif Ma > 1:
		return x_post_col(A_Astar)
	else:
		return X_star

# Discretisation
n = int(1e4)
dx = (X_points[-1] - X_points[0]) / n