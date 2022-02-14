import numpy as	np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from cnst import DATA_BASENAME, GAMMA, C_p, C_v, R, P_0, T_0, X_points, H_points, H_star, A_star, A_Astar, \
	q_points, q_func, n, dx
from iso_properties import area_ratio_Ma, T_ratio_Ma, p_ratio_Ma, speed_of_sound_Ma, speed_of_sound_T, x_Ma


def dM2_heat(M, dA_A, q, T, cp=C_p, gamma=GAMMA):
	return M ** 2 * (dA_A * -2 * (1 + (gamma - 1)/2 * M ** 2) / (1 - M ** 2)
		+ q / cp / T * (1 + gamma * M ** 2)/(1 - M ** 2))

def dT_heat(M, dA_A, q, T, cp=C_p, gamma=GAMMA):
	return T * (dA_A * (gamma - 1) * M ** 2 / (1 - M ** 2) +
			q / cp / T * (1 - gamma * M ** 2) / (1 - M ** 2))

def drho_heat(M, dA_A, q, T, rho, cp=C_p, gamma=GAMMA):
	return rho * (dA_A * M ** 2 / (1 - M**2) - q / cp / T / (1 - M ** 2))

func = lambda mach: area_ratio_Ma(mach) - H_points[-1] / H_star
Ma_max = fsolve(func, 2)
func1 = lambda mach: area_ratio_Ma(mach) - H_points[0] / H_star
Ma_min = fsolve(func1, 0.1)
print(Ma_max)
print(Ma_min)
Ma_lin = np.linspace(Ma_min + 0.001, Ma_max - 0.001, n)
X = np.array([x_Ma(Ma) for Ma in Ma_lin])

# Mass flow, choked flow
m_dot = P_0 / p_ratio_Ma(1) * np.sqrt(GAMMA / (R * T_0 / T_ratio_Ma(1))) * A_star

# Déterminer début de la condensation
Ma0_cond = 1.001  # nombre de Mach ou la condensation début.

# position et conditions au début de la condensation
x0_cond = x_Ma(Ma0_cond)
# Uncomment for  Rayleigh flow
# x0_cond = 0.17
func2 =lambda mach: area_ratio_Ma(mach) - A_Astar(x0_cond)
Ma0_cond = fsolve(func2, 3)
u0_cond = Ma0_cond * speed_of_sound_Ma(Ma0_cond)
p0_cond = P_0 / p_ratio_Ma(Ma0_cond) 
T0_cond = T_0 / T_ratio_Ma(Ma0_cond)
rho0_cond = p0_cond / (R * T0_cond)
print(x0_cond)
print("---------CONDITIONS INITIALES-----------")
print("Mach:")
print(Ma0_cond)
print("pression:")
print(p0_cond)
print("Température:")
print(T0_cond)
print("densité:")
print(rho0_cond)

# Discrétisation de la condensation
print(X_points[-1], dx, x0_cond)
x_cond = np.arange(x0_cond, q_points[0, -1] / 100, dx)
A_cond = A_star * A_Astar(x_cond)

Ma_cond = np.zeros(x_cond.shape)
Ma_cond[0] = Ma0_cond

u_cond = np.zeros(x_cond.shape)
u_cond[0] = u0_cond

p_cond = np.zeros(x_cond.shape)
p_cond[0] = p0_cond

T_cond = np.zeros(x_cond.shape)
T_cond[0] = T0_cond

# T_stag_cond = np.zeros(x_cond.shape)
# T_stag_cond[0] = T_0

rho_cond = np.zeros(x_cond.shape)
rho_cond[0] = rho0_cond

# Solve écoulement
for i in range(len(x_cond)-1):
	# if (x_cond[i]-x_cond[0]) < COND_LEN:
	# 	q = dqdx * (x_cond[i+1]-x_cond[i])
	# else:
	# 	q = 0
	q = q_func(x_cond[i + 1]) - q_func(x_cond[i])
	# T_stag_cond[i+1] = q/C_p + T_stag_cond[i]
	Ma_cond[i+1] = np.sqrt(Ma_cond[i]**2 + dM2_heat(Ma_cond[i], 
		(A_cond[i+1]-A_cond[i])/A_cond[i], q, T_cond[i]))
	T_cond[i+1] = T_cond[i] + dT_heat(Ma_cond[i], 
		(A_cond[i+1]-A_cond[i])/A_cond[i], q, T_cond[i])

	u_cond[i+1] = Ma_cond[i+1] * speed_of_sound_T(T_cond[i + 1])
	rho_cond[i+1] = rho_cond[i] + drho_heat(Ma_cond[i], 
		(A_cond[i+1]-A_cond[i])/A_cond[i], q, T_cond[i], rho_cond[i])

# Calculate pressure:
p_cond = rho_cond * R * T_cond

print("---------CONDITIONS FINALES-------------")
print("Mach:")
print(Ma_cond[-1])
print("pression:")
print(p_cond[-1])
print("Température:")
print(T_cond[-1])
print("densité:")
print(rho_cond[-1])

# conservation checks
print("Masse (kg/s)")
print(rho_cond[0]*u_cond[0]*A_cond[0] - rho_cond[-1]*u_cond[-1]*A_cond[-1])
print("Energy (J/kg)")
print(C_p * T_cond[0] + u_cond[0] ** 2 / 2 - C_p * T_cond[-1] - u_cond[-1] ** 2 / 2)
print("Momentum (N)")
print(u_cond[0] * m_dot - u_cond[-1] * m_dot 
	+ sum(p_cond[1:] * (A_cond[1:]-A_cond[:-1])) 
	+ p_cond[0]*A_cond[0] 
	- p_cond[-1]*A_cond[-1])
print("entropy (J/kgK)")
print(C_v * np.log(T_cond[-1] / T_cond[0]) - R * np.log(rho_cond[-1] / rho_cond[0]))
	

#-----------------------------------------------------------------------------
#                               PLOTTING
#-----------------------------------------------------------------------------
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)
ax0.plot(X, P_0 / p_ratio_Ma(Ma_lin))
ax0.plot(x_cond, p_cond)
ax0.set_xlabel('x (m)')
ax0.set_ylabel('P (Pa)')
ax1.plot(X, T_0 / T_ratio_Ma(Ma_lin))
ax1.plot(x_cond, T_cond)
ax1.set_xlabel('x (m)')
ax1.set_ylabel('T (K)')
ax2.plot(X, Ma_lin)
ax2.plot(x_cond, Ma_cond)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('Mach')
# ax2.plot(X, rho_0/rho_ratio_Ma(Ma_lin))
# ax2.plot(x_cond, rho_cond)
# ax2.set_xlabel('x (m)')
# ax2.set_ylabel('rho (kg/m³)')
ax3.plot(X, speed_of_sound_Ma(Ma_lin) * Ma_lin)
ax3.plot(x_cond, u_cond)
ax3.set_xlabel('x (m)')
ax3.set_ylabel('u (m/s)')
# plt.figure()
# plt.plot(x_cond, q_func(x_cond))

f = plt.figure()
ax = f.add_subplot(111)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
im = plt.imread(DATA_BASENAME + ".PT.png")
implot = ax.imshow(im, origin="upper", extent=(0,12,100,240), aspect='auto')
ax.plot(X * 100, T_0 / T_ratio_Ma(Ma_lin))
ax.plot(x_cond*100, T_cond)
ax.set_xlim([0,12])
ax.set_ylim([100,240])



plt.show()



