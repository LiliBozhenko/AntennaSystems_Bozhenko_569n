import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ===== Варіант 2 (λ=2.6 см, a_p=b_p=11 см)
lam = 0.026
ap = 0.11
bp = 0.11

# Константи для 3.41 (H) і 3.42 (E)
C = np.pi * ap / lam
D = 2 * ap / lam
Ce, Ch = C, C
De, Dh = D, D

# ---- частини формул 3.42 (E) та 3.41 (H)
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

# ---- допоміжні: інтерполяції
def interp_x_at_y(x, y, y0):
    i = np.where((y[:-1]-y0)*(y[1:]-y0) <= 0)[0]
    xs = []
    for k in i:
        x1, x2 = x[k], x[k+1]
        y1, y2 = y[k], y[k+1]
        if y2 == y1:
            xs.append(x1)
        else:
            xs.append(x1 + (y0 - y1) * (x2 - x1) / (y2 - y1))
    return np.array(xs)

def hpbw_deg(x_deg, F):
    xs = np.deg2rad(x_deg)
    th_cross = interp_x_at_y(xs, F, 0.707)
    if len(th_cross) < 2:
        return np.nan, np.nan, np.nan
    t1 = th_cross.min()
    t2 = th_cross.max()
    return np.rad2deg(t2 - t1), np.rad2deg(t1), np.rad2deg(t2)

def first_sidelobe(x_deg, F, right_of_deg):
    mask = x_deg > right_of_deg
    idx, _ = find_peaks(F[mask])
    if len(idx)==0:
        return None
    th = x_deg[mask][idx][0]
    val = F[mask][idx][0]
    return th, val, 20*np.log10(val + 1e-12)

def nulls_deg(x_deg, F):
    xs = np.deg2rad(x_deg)
    z = interp_x_at_y(xs, F, 0.0)
    return np.rad2deg(z)

# ---- сітки кутів
deg_full = np.linspace(-90.0, 90.0, 18001)
th_full  = np.deg2rad(deg_full)

deg = np.linspace(0.0, 90.0, 9001)      # для відображення
th  = np.deg2rad(deg)

# ---- повні ДС на [-90..90] (для метрик)
Fe_full = Fce(th_full)*F1e(th_full); Fe_full /= Fe_full.max()
Fh_full = Fch(th_full)*F1h(th_full); Fh_full /= Fh_full.max()

# ---- метрики з повної ДС
HP_E, t1E, t2E = hpbw_deg(deg_full, Fe_full)
HP_H, t1H, t2H = hpbw_deg(deg_full, Fh_full)
nullE_full = nulls_deg(deg_full, Fe_full)
nullH_full = nulls_deg(deg_full, Fh_full)

# перші бічні після правої межі HPBW (симетрія)
slE = first_sidelobe(deg_full, Fe_full, t2E if np.isfinite(t2E) else 0)
slH = first_sidelobe(deg_full, Fh_full, t2H if np.isfinite(t2H) else 0)

# ---- криві для відображення [0..90]
Fe = Fce(th)*F1e(th); Fe /= Fe.max()
Fh = Fch(th)*F1h(th); Fh /= Fh.max()

# нулі в передній напівплощині
nullE = nullE_full[(nullE_full >= 0) & (nullE_full <= 90)]
nullH = nullH_full[(nullH_full >= 0) & (nullH_full <= 90)]

# ---- графік (лінійний)
plt.figure(figsize=(7.6, 4.4))
plt.plot(deg, Fh, label="H-площина (3.41)")
plt.plot(deg, Fe, '--', label="E-площина (3.42)")
plt.axhline(0.707, color='gray', ls='--', lw=0.8)

# межі HPBW (беремо праві половини для візу), показуємо як тонкі лінії
if np.isfinite(t2H): plt.axvline(t2H, color='C0', ls=':', lw=0.9)
if np.isfinite(t2E): plt.axvline(t2E, color='C1', ls=':', lw=0.9)

if len(nullH): plt.scatter(nullH, np.zeros_like(nullH), c='C0', s=18, zorder=3, label="Нулі H")
if len(nullE): plt.scatter(nullE, np.zeros_like(nullE), c='C1', s=18, zorder=3, label="Нулі E")

if slH and 0 <= slH[0] <= 90: plt.scatter([slH[0]], [slH[1]], c='C0', s=30, zorder=3, label="Бічна H (ампл.)")
if slE and 0 <= slE[0] <= 90: plt.scatter([slE[0]], [slE[1]], c='C1', s=30, zorder=3, label="Бічна E (ампл.)")

plt.title("Нормовані ДС пірамідального рупора (лінійний масштаб)")
plt.xlabel("θ, град"); plt.ylabel("Амплітуда (норма на максимум)")
plt.xlim(0, 90); plt.ylim(0, 1.05); plt.grid(ls='--', alpha=0.3); plt.legend(loc="upper right")

# ---- графік (дБ)
plt.figure(figsize=(7.6, 4.4))
plt.plot(deg, 20*np.log10(Fh+1e-12), label="H-площина (3.41)")
plt.plot(deg, 20*np.log10(Fe+1e-12), '--', label="E-площина (3.42)")
if slH and 0 <= slH[0] <= 90: plt.scatter([slH[0]], [slH[2]], c='C0', s=30, zorder=3, label=f"1-ша бічна H: {slH[2]:.1f} дБ")
if slE and 0 <= slE[0] <= 90: plt.scatter([slE[0]], [slE[2]], c='C1', s=30, zorder=3, label=f"1-ша бічна E: {slE[2]:.1f} дБ")
plt.title("Нормовані ДС пірамідального рупора (дБ)")
plt.xlabel("θ, град"); plt.ylabel("Рівень, дБ (норма на максимум)")
plt.xlim(0, 90); plt.ylim(-60, 0); plt.grid(ls='--', alpha=0.3); plt.legend(loc="upper right")

# ---- друк чисел
print(f"λ = {lam:.3f} м, a_p = {ap:.3f} м, b_p = {bp:.3f} м")
print(f"πa_p/λ = πb_p/λ = {C:.4f},  2a_p/λ = 2b_p/λ = {D:.4f}")
print(f"HPBW_E ≈ {HP_E:.2f}°  (межі {t1E:.2f}° .. {t2E:.2f}°)")
print(f"HPBW_H ≈ {HP_H:.2f}°  (межі {t1H:.2f}° .. {t2H:.2f}°)")
if slE: print(f"1-ша бічна (E): ампл={slE[1]:.3f}, рівень={slE[2]:.2f} дБ @ θ≈{slE[0]:.2f}°")
if slH: print(f"1-ша бічна (H): ампл={slH[1]:.3f}, рівень={slH[2]:.2f} дБ @ θ≈{slH[0]:.2f}°")

plt.show()
