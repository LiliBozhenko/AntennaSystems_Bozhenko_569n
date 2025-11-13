import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

lam = 0.026
ap = 0.11
bp = 0.11

C = np.pi * ap / lam
D = 2 * ap / lam
Ce, Ch = C, C
De, Dh = D, D

deg = np.linspace(0.0, 90.0, 9001)
th  = np.deg2rad(deg)

def Fce(theta):
    x = Ce * np.sin(theta)
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = np.sin(x[nz]) / x[nz]
    return np.abs(out)

def F1e(theta):
    return (1 + np.cos(theta)) / 2

def Fch(theta):
    x = Ch * np.sin(theta)
    denom = 1 - (Dh * np.sin(theta))**2
    return np.abs(np.cos(x) / denom)

def F1h(theta):
    return (1 + np.cos(theta)) / 2

Fe = Fce(th) * F1e(th)
Fh = Fch(th) * F1h(th)

Fe /= Fe.max()
Fh /= Fh.max()

def hpbw(theta, F):
    lvl = 0.707
    sgn = np.sign(F - lvl)
    cross = np.where(np.diff(sgn) != 0)[0]
    if len(cross) >= 2:
        t1, t2 = theta[cross[0]], theta[cross[1]]
        return np.rad2deg(t2 - t1), np.rad2deg(t1), np.rad2deg(t2)
    return np.nan, np.nan, np.nan

HP_E, t1E, t2E = hpbw(th, Fe)
HP_H, t1H, t2H = hpbw(th, Fh)

def sidelobes(theta_deg, F, right_of_deg):
    mask = theta_deg > right_of_deg
    idx, _ = find_peaks(F[mask])
    th_pe = theta_deg[mask][idx]
    val_pe = F[mask][idx]
    if len(val_pe) == 0:
        return None
    return th_pe[0], 20*np.log10(val_pe[0] + 1e-12)

slE = sidelobes(deg, Fe, t2E if np.isfinite(t2E) else 0)
slH = sidelobes(deg, Fh, t2H if np.isfinite(t2H) else 0)

plt.figure(figsize=(7.2, 4.2))
plt.plot(deg, Fh, label="H-площина (3.41)")
plt.plot(deg, Fe, '--', label="E-площина (3.42)")
plt.axhline(0.707, color='gray', ls='--', lw=0.8)
for t in [t1H, t2H]:
    if np.isfinite(t): plt.axvline(t, color='C0', ls=':', lw=0.8)
for t in [t1E, t2E]:
    if np.isfinite(t): plt.axvline(t, color='C1', ls=':', lw=0.8)
plt.title("Нормовані ДС пірамідального рупора (лінійний масштаб)")
plt.xlabel("θ, град"); plt.ylabel("Амплітуда (норма на максимум)")
plt.xlim(0, 90); plt.ylim(0, 1.05); plt.grid(ls='--', alpha=0.3); plt.legend()

plt.figure(figsize=(7.2, 4.2))
plt.plot(deg, 20*np.log10(Fh+1e-12), label="H-площина (3.41)")
plt.plot(deg, 20*np.log10(Fe+1e-12), '--', label="E-площина (3.42)")
if slH: plt.scatter([slH[0]], [slH[1]], c='C0', s=25, label=f"1-ша бічна H: {slH[1]:.1f} дБ")
if slE: plt.scatter([slE[0]], [slE[1]], c='C1', s=25, label=f"1-ша бічна E: {slE[1]:.1f} дБ")
plt.title("Нормовані ДС пірамідального рупора (дБ)")
plt.xlabel("θ, град"); plt.ylabel("Рівень, дБ (норма на максимум)")
plt.xlim(0, 90); plt.ylim(-60, 0); plt.grid(ls='--', alpha=0.3); plt.legend()

print(f"λ = {lam:.3f} м, a_p = {ap:.3f} м, b_p = {bp:.3f} м")
print(f"Константи: πa_p/λ = πb_p/λ = {C:.4f},  2a_p/λ = 2b_p/λ = {D:.4f}")
print(f"HPBW_E ≈ {HP_E:.2f}°  (межі {t1E:.2f}° .. {t2E:.2f}°)")
print(f"HPBW_H ≈ {HP_H:.2f}°  (межі {t1H:.2f}° .. {t2H:.2f}°)")
if slE: print(f"Найближча бічна пелюстка (E): {slE[1]:.2f} дБ @ θ≈{slE[0]:.2f}°")
if slH: print(f"Найближча бічна пелюстка (H): {slH[1]:.2f} дБ @ θ≈{slH[0]:.2f}°")

plt.show()
