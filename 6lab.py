import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.special import j1

lambda_cm = 3.2
D_m = 1.0
f_m = 0.45

lam = lambda_cm/100.0
k = 2*np.pi/lam
a = D_m/2.0

def airy_field(theta):
    x = k*a*np.sin(theta)
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = 2.0*j1(x[nz])/x[nz]
    return np.abs(out)

def FE(theta): return airy_field(theta)
def FH(theta): return airy_field(theta)

deg_full = np.linspace(-90.0, 90.0, 18001)
th_full  = np.deg2rad(deg_full)
deg = np.linspace(0.0, 90.0, 9001)
th  = np.deg2rad(deg)

FE_full = FE(th_full); FE_full /= FE_full.max()
FH_full = FH(th_full); FH_full /= FH_full.max()
FE_disp = FE(th); FE_disp /= FE_disp.max()
FH_disp = FH(th); FH_disp /= FH_disp.max()

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

HP_E, t1E, t2E = hpbw_deg(deg_full, FE_full)
HP_H, t1H, t2H = hpbw_deg(deg_full, FH_full)
nullE_full = nulls_deg(deg_full, FE_full)
nullH_full = nulls_deg(deg_full, FH_full)
slE = first_sidelobe(deg_full, FE_full, t2E if np.isfinite(t2E) else 0)
slH = first_sidelobe(deg_full, FH_full, t2H if np.isfinite(t2H) else 0)

nullE = nullE_full[(nullE_full >= 0) & (nullE_full <= 90)]
nullH = nullH_full[(nullH_full >= 0) & (nullH_full <= 90)]

plt.figure(figsize=(7.6, 4.4))
plt.plot(deg, FH_disp, label="H-площина (кругла апертура)")
plt.plot(deg, FE_disp, '--', label="E-площина (кругла апертура)")
plt.axhline(0.707, color='gray', ls='--', lw=0.8)
if np.isfinite(t2H): plt.axvline(t2H, color='C0', ls=':', lw=0.9)
if np.isfinite(t2E): plt.axvline(t2E, color='C1', ls=':', lw=0.9)
if len(nullH): plt.scatter(nullH, np.zeros_like(nullH), c='C0', s=18, zorder=3, label="Нулі H")
if len(nullE): plt.scatter(nullE, np.zeros_like(nullE), c='C1', s=18, zorder=3, label="Нулі E")
if slH and 0 <= slH[0] <= 90: plt.scatter([slH[0]], [slH[1]], c='C0', s=30, zorder=3, label="Бічна H")
if slE and 0 <= slE[0] <= 90: plt.scatter([slE[0]], [slE[1]], c='C1', s=30, zorder=3, label="Бічна E")
plt.title("Нормовані ДС дзеркальної антени — лінійний масштаб")
plt.xlabel("θ, град"); plt.ylabel("Амплітуда (норма на максимум)")
plt.xlim(0, 90); plt.ylim(0, 1.05); plt.grid(ls='--', alpha=0.3); plt.legend(loc="upper right")

plt.figure(figsize=(7.6, 4.4))
plt.plot(deg, 20*np.log10(FH_disp+1e-12), label="H-площина")
plt.plot(deg, 20*np.log10(FE_disp+1e-12), '--', label="E-площина")
if slH and 0 <= slH[0] <= 90: plt.scatter([slH[0]], [20*np.log10(slH[1]+1e-12)], c='C0', s=30, zorder=3, label="1-ша бічна H")
if slE and 0 <= slE[0] <= 90: plt.scatter([slE[0]], [20*np.log10(slE[1]+1e-12)], c='C1', s=30, zorder=3, label="1-ша бічна E")
plt.title("Нормовані ДС дзеркальної антени — дБ")
plt.xlabel("θ, град"); plt.ylabel("Рівень, дБ (норма на максимум)")
plt.xlim(0, 90); plt.ylim(-60, 0); plt.grid(ls='--', alpha=0.3); plt.legend(loc="upper right")

print(f"λ = {lam:.3f} м (λ={lambda_cm} см), D = {D_m:.2f} м, f = {f_m:.2f} м")
print(f"HPBW_E ≈ {HP_E:.2f}°  (межі {t1E:.2f}° .. {t2E:.2f}°)")
print(f"HPBW_H ≈ {HP_H:.2f}°  (межі {t1H:.2f}° .. {t2H:.2f}°)")
if slE: print(f"1-ша бічна (E): ампл={slE[1]:.3f}, рівень={20*np.log10(slE[1]+1e-12):.2f} дБ @ θ≈{slE[0]:.2f}°")
if slH: print(f"1-ша бічна (H): ампл={slH[1]:.3f}, рівень={20*np.log10(slH[1]+1e-12):.2f} дБ @ θ≈{slH[0]:.2f}°")

plt.show()
