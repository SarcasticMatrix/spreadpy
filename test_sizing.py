import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def kelly_fraction(z0, z_revert, f_max=0.25):
    phi = norm.pdf(z0)
    Phi = norm.cdf(z0)
    lam = phi / (1 - Phi)
    
    E_G = z0 + lam - z_revert
    Var_z = 1 - lam * (lam - z0)
    E_G2 = Var_z + E_G**2
    
    if E_G <= 0 or E_G2 <= 0:
        return 0.0
    
    return min(E_G / E_G2, f_max)

# Grille
z0_vals   = np.linspace(0.5, 3.0, 200)
zrev_vals = np.linspace(-1.5, 1.5, 200)
Z0, ZREV  = np.meshgrid(z0_vals, zrev_vals)

F = np.vectorize(kelly_fraction)(Z0, ZREV, f_max=0.25)

# Plot 3D
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(Z0, ZREV, F, cmap="viridis", edgecolor="none", alpha=0.92)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="$f^*$", pad=0.1)

ax.set_xlabel("$z_0$  (entry threshold)", fontsize=11, labelpad=10)
ax.set_ylabel("$z_{\\mathrm{revert}}$  (reversion target)", fontsize=11, labelpad=10)
ax.set_zlabel("$f^*$  (Kelly fraction)", fontsize=11, labelpad=8)
ax.set_title("Kelly fraction $f^*$", fontsize=13, pad=15)

# Slice z_revert = 0 en rouge
f_slice = np.array([kelly_fraction(z, 0.0) for z in z0_vals])
ax.plot(z0_vals, np.zeros_like(z0_vals), f_slice,
        color="red", linewidth=2, label="$z_{\\mathrm{revert}}=0$")

ax.legend(fontsize=10)
ax.view_init(elev=30, azim=-60)

plt.tight_layout()
plt.show()