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
Me = 2.68
gamma = 1.35
wall, characteristics, left_runn_chars, theta_w_max = min_length_supersonic_nozzle_moc(ht, n, Me, None, gamma)
x, y, z = np.array([]), np.array([]), np.array([])
for char in left_runn_chars:
    x = np.append(x, char["x"])
    y = np.append(y, char["y"])
    z = np.append(z, char["M"])
plt.figure()
for c in characteristics:
    plt.plot(c["x"], c["y"], "k:", linewidth=0.65)
    plt.plot(wall[:, 0], wall[:, 1], "k")
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
plt.show()
