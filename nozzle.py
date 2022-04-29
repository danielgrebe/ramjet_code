import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from pygasflow.nozzles.moc import min_length_supersonic_nozzle_moc
from pygasflow.nozzles import (
    CD_Conical_Nozzle,
    CD_TOP_Nozzle,
    CD_Min_Length_Nozzle
)
ht = 0.5
n = 20
Me = 2.72
gamma = 1.35
wall, characteristics, left_runn_chars, theta_w_max = min_length_supersonic_nozzle_moc(ht, n, Me, None, gamma)
x, y, z = np.array([]), np.array([]), np.array([])
for char in left_runn_chars:
    x = np.append(x, char["x"])
    y = np.append(y, char["y"])
    z = np.append(z, char["M"])
plt.figure()
# draw characteristics lines
for c in characteristics:
    plt.plot(c["x"], c["y"], "k:", linewidth=0.65)
    # draw nozzle wall
    plt.plot(wall[:, 0], wall[:, 1], "k")
# over impose grid points colored by Mach number
sc = plt.scatter(x, y, c=z, s=15, vmin=min(z), vmax=max(z), cmap=cmx.cool)
cbar = plt.colorbar(sc, orientation='vertical', aspect=40)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel("Mach number", rotation=270)
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"$M_e$ = {}, n = {}, ht = {} ".format(Me, n, ht))
plt.grid()
plt.axis('equal')
plt.tight_layout()

# inlet radius of the convergent section
Ri = 14.74
# throat radius
Rt = 14.61
# exit (outlet) radius of the divergent section
Re = 53.15
# junction radius between the convergent and divergent at the throat section. Used in the conical nozzle.
Rj = 0.1
# junction radius between the combustion chamber and the convergent
R0 = 0.2
# half cone angle of the convergent section
theta_c = 40
# half cone angle of the divergent section
theta_N = 15
# fractional lengths. Used to construct TOP nozzles.
K = [0.6, 0.7, 0.8, 0.9, 1]

geom_type = "planar"

conical = CD_Conical_Nozzle(Ri, Re, Rt, Rj, R0, theta_c, theta_N, geom_type)

# fractional length of the TOP nozzle
K = 0.7
#top = CD_TOP_Nozzle(Ri, Re, Rt, R0, theta_c, 0.7, geom_type)

# number of characteristics
n = 20
# specific heats ratio
gamma = 1.32

moc = CD_Min_Length_Nozzle(Ri, Re, Rt, Rj, R0, theta_c, n, gamma)

#plt.figure()
N = 1000
#x1, y1 = conical.build_geometry(N)
#x2, y2 = top.build_geometry(N)
x3, y3 = moc.build_geometry(N)
plt.figure()
#plt.plot(x1, y1, label="conical")
#plt.plot(x2, y2, label="TOP: K = {}".format(top.Fractional_Length))
plt.plot(x3, y3, label="MOC")
plt.legend()
plt.xlabel("Length")
plt.ylabel("Half Height")
plt.title("Planar")
plt.minorticks_on()
plt.grid(which='major', linestyle='-', alpha=0.7)
plt.grid(which='minor', linestyle=':', alpha=0.5)
plt.axis('equal')
plt.show()

