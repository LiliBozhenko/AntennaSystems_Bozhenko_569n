import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

lambda_cm = 3.3
lam = lambda_cm/100.0
k = 2*np.pi/lam

N = 5
Ls_cm = 2.0
d_cm  = 4.0
Ls = Ls_cm/100.0
d  = d_cm/100.0

psi = 0.0

def sinc(x):
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = np.sin(x[nz])/x[nz]
    return out

def slot_E_factor(theta):
    u = 0.5*k*Ls*np.sin(theta)
    return np.abs(sinc(u) * np.sin(theta))

def slot_H_factor(theta):
    u = 0.5*k*Ls*np.sin(theta)
    return np.abs(sinc(u) * np.cos(theta))

def array_factor(theta):
    kd = k*d*np.sin(theta) + psi
    x = kd
    num = np.sin(N*x/2.0)
    den = np.sin(x/2.0)
    AF = np.ones_like(theta, dtype=float)
    nz = np.abs(den) > 1e-12
    AF[nz] = np.abs(num[nz]/den[nz])
    AF[~nz] = N
    return AF / N

def pattern_E(theta):
    return slot_E_factor(theta) * array_factor(theta)

def pattern_H(theta):
    return slot_H_factor(theta) * array_factor(theta)

def interp_x_at_y(x, y, y0):
    idx = np.where((y[:-1]-y0)*(y[1:]-y0) <= 0)[0]
    xs = []
    for i in idx:
        x1, x2 = x[i], x[i+1]
        y1, y2 = y[i], y[i+1]
        xs.append(x1 if y2==y1 else x1 + (y0 - y1) * (x2 - x1) / (y2 - y1))
    return np.array(xs)

def hpbw_deg(x_deg, F):
    xs = np.deg2rad(x_deg)
    cross = interp_x_at_y(xs, F, 0.707)
    if len(cross) < 2: return np.nan, np.nan, np.nan
    t1, t2 = cross.min(), cross.max()
    return np.rad2deg(t2 - t1), np.rad2deg(t1), np.rad2deg(t2)

def first_sidelobe(x_deg, F, right_of_deg):
    mask = x_deg > right_of_deg
    idx, _ = find_peaks(F[mask])
    if len(idx)==0: return None
    th = x_deg[mask][idx][0]
    val = F[mask][idx][0]
    return th, val, 20*np.log10(val + 1e-12)

def nulls_deg(x_deg, F):
    xs = np.deg2rad(x_deg)
    z = interp_x_at_y(xs, F, 0.0)
    return np.rad2deg(z)

deg_full = np.linspace(-90.0, 90.0, 18001)
th_full  = np.deg2rad(deg_full)

deg = np.linspace(0.0, 90.0, 9001)
th  = np.deg2rad(deg)

FE_full = pattern_E(th_full); FE_full /= FE_full.max()
FH_full = pattern_H(th_full); FH_full /= FH_full.max()

FE = pattern_E(th); FE /= FE.max()
FH = pattern_H(th); FH /= FH.max()

HP_E, t1E, t2E = hpbw_deg(deg_full, FE_full)
HP_H, t1H, t2H = hpbw_deg(deg_full, FH_full)
nullE_full = nulls_deg(deg_full, FE_full)
nullH_full = nulls_deg(deg_full, FH_full)
slE = first_sidelobe(deg_full, FE_full, t2E if np.isfinite(t2E) else 0)
slH = first_sidelobe(deg_full, FH_full, t2H if np.isfinite(t2H) else 0)

nullE = nullE_full[(nullE_full >= 0) & (nullE_full <= 90)]
nullH = nullH_full[(nullH_full >= 0) & (nullH_full <= 90)]

plt.figure(figsize=(7.6, 4.4))
plt.plot(deg, FH, label="H-площина (тип 2)")
plt.plot(deg, FE, '--', label="E-площина (тип 2)")
plt.axhline(0.707, color='gray', ls='--', lw=0.8)
if np.isfinite(t2H): plt.axvline(t2H, color='C0', ls=':', lw=0.9)
if np.isfinite(t2E): plt.axvline(t2E, color='C1', ls=':', lw=0.9)
if len(nullH): plt.scatter(nullH, np.zeros_like(nullH), c='C0', s=18, zorder=3, label="Нулі H")
if len(nullE): plt.scatter(nullE, np.zeros_like(nullE), c='C1', s=18, zorder=3, label="Нулі E")
if slH and 0 <= slH[0] <= 90: plt.scatter([slH[0]], [slH[1]], c='C0', s=30, zorder=3, label="Бічна H")
if slE and 0 <= slE[0] <= 90: plt.scatter([slE[0]], [slE[1]], c='C1', s=30, zorder=3, label="Бічна E")
plt.title("Нормовані ДС хвилеводно-щілинної антени (тип 2) — лінійний масштаб")
plt.xlabel("θ, град"); plt.ylabel("Амплітуда (норма на максимум)")
plt.xlim(0, 90); plt.ylim(0, 1.05); plt.grid(ls='--', alpha=0.3); plt.legend(loc="upper right")

plt.figure(figsize=(7.6, 4.4))
plt.plot(deg, 20*np.log10(FH+1e-12), label="H-площина (тип 2)")
plt.plot(deg, 20*np.log10(FE+1e-12), '--', label="E-площина (тип 2)")
if slH and 0 <= slH[0] <= 90: plt.scatter([slH[0]], [20*np.log10(slH[1]+1e-12)], c='C0', s=30, zorder=3, label="1-ша бічна H")
if slE and 0 <= slE[0] <= 90: plt.scatter([slE[0]], [20*np.log10(slE[1]+1e-12)], c='C1', s=30, zorder=3, label="1-ша бічна E")
plt.title("Нормовані ДС хвилеводно-щілинної антени (тип 2) — дБ")
plt.xlabel("θ, град"); plt.ylabel("Рівень, дБ (норма на максимум)")
plt.xlim(0, 90); plt.ylim(-60, 0); plt.grid(ls='--', alpha=0.3); plt.legend(loc="upper right")

print(f"λ = {lam:.3f} м  (λ={lambda_cm} см)")
print(f"Тип 2: N={N}, Ls={Ls_cm} см, d={d_cm} см, ψ={psi:.2f} рад")
print(f"HPBW_E ≈ {HP_E:.2f}°  (межі {t1E:.2f}° .. {t2E:.2f}°)")
print(f"HPBW_H ≈ {HP_H:.2f}°  (межі {t1H:.2f}° .. {t2H:.2f}°)")
if slE: print(f"1-ша бічна (E): ампл={slE[1]:.3f}, рівень={20*np.log10(slE[1]+1e-12):.2f} дБ @ θ≈{slE[0]:.2f}°")
if slH: print(f"1-ша бічна (H): ампл={slH[1]:.3f}, рівень={20*np.log10(slH[1]+1e-12):.2f} дБ @ θ≈{slH[0]:.2f}°")

plt.show()
