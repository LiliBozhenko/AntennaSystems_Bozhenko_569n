import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.special import j1

# ----- Варіант 2: вихідні дані -----
lambda_cm = 3.2
h_cm      = 8.0
xi        = 1.10
d_avg_cm  = 2.3
L_cm      = 23.7

lam = lambda_cm/100.0
h   = h_cm/100.0
d_eff = d_avg_cm/100.0
L     = L_cm/100.0
k = 2*np.pi/lam

def Lambda1(x):
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = 2.0*j1(x[nz])/x[nz]
    return out

# (4.3) без модуля
def F_b(theta):
    num1 = (xi - 1.0) / np.sin(np.pi*L/lam * (xi - 1.0))
    term = np.sin(np.pi*L/lam * (xi - np.cos(theta))) / (xi - np.cos(theta))
    return num1 * term

# (4.4), (4.5)
def F_1E(theta):
    x = np.pi*d_eff/lam * np.sin(theta)
    return Lambda1(x) * np.cos(theta)

def F_1H(theta):
    x = np.pi*d_eff/lam * np.sin(theta)
    return Lambda1(x)

# (4.2)
def FE_one(theta): return F_b(theta) * F_1E(theta)
def FH_one(theta): return F_b(theta) * F_1H(theta)

# (4.16): SIN-варіант, як у вашому «правильному» графіку
def Fc(theta): return 2.0*np.cos(0.5*k*h*np.sin(theta))
def FE_two(theta): return FE_one(theta) * Fc(theta)
def FH_two(theta): return FH_one(theta) * Fc(theta)

# сітки
deg_full = np.linspace(-90.0, 90.0, 36001)
th_full  = np.deg2rad(deg_full)
deg = np.linspace(0.0, 90.0, 18001)
th  = np.deg2rad(deg)

def patt_norm(f, theta):
    y = np.abs(f(theta))
    return y / y.max() if y.max()!=0 else y

# повні криві
Fe1_full = patt_norm(FE_one, th_full)
Fh1_full = patt_norm(FH_one, th_full)
Fe2_full = patt_norm(FE_two, th_full)
Fh2_full = patt_norm(FH_two, th_full)

# для відображення
Fe1 = patt_norm(FE_one, th)
Fh1 = patt_norm(FH_one, th)
Fe2 = patt_norm(FE_two, th)
Fh2 = patt_norm(FH_two, th)

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

def all_nulls_deg(x_deg, F):
    xs = np.deg2rad(x_deg)
    z = interp_x_at_y(xs, F, 0.0)
    return np.rad2deg(z[(z>=0)&(z<=np.pi/2)])

def all_sidelobes(x_deg, F, right_of_deg=0.0):
    mask = x_deg > right_of_deg
    idx, _ = find_peaks(F[mask])
    th = x_deg[mask][idx]
    val = F[mask][idx]
    return th, val, 20*np.log10(val + 1e-12)

HP_E1, t1E1, t2E1 = hpbw_deg(deg_full, Fe1_full)
HP_H1, t1H1, t2H1 = hpbw_deg(deg_full, Fh1_full)
HP_E2, t1E2, t2E2 = hpbw_deg(deg_full, Fe2_full)
HP_H2, t1H2, t2H2 = hpbw_deg(deg_full, Fh2_full)

nullE1 = all_nulls_deg(deg_full, Fe1_full)
nullH1 = all_nulls_deg(deg_full, Fh1_full)
nullE2 = all_nulls_deg(deg_full, Fe2_full)
nullH2 = all_nulls_deg(deg_full, Fh2_full)

thSL_E1, SL_E1, SLdB_E1 = all_sidelobes(deg_full, Fe1_full, t2E1 if np.isfinite(t2E1) else 0)
thSL_H1, SL_H1, SLdB_H1 = all_sidelobes(deg_full, Fh1_full, t2H1 if np.isfinite(t2H1) else 0)
thSL_E2, SL_E2, SLdB_E2 = all_sidelobes(deg_full, Fe2_full, t2E2 if np.isfinite(t2E2) else 0)
thSL_H2, SL_H2, SLdB_H2 = all_sidelobes(deg_full, Fh2_full, t2H2 if np.isfinite(t2H2) else 0)

plt.figure(figsize=(9,5.1))
plt.plot(deg, Fh1, label="H, 1-стриж.")
plt.plot(deg, Fe1, '--', label="E, 1-стриж.")
plt.plot(deg, Fh2, label="H, 2-стриж.")
plt.plot(deg, Fe2, '--', label="E, 2-стриж.")
plt.axhline(0.707, color='gray', ls='--', lw=0.8)
for z in nullH1[(nullH1>=0)&(nullH1<=90)]: plt.axvline(z, color='C0', ls='--', lw=0.6, alpha=0.45)
for z in nullE1[(nullE1>=0)&(nullE1<=90)]: plt.axvline(z, color='C1', ls='--', lw=0.6, alpha=0.45)
for z in nullH2[(nullH2>=0)&(nullH2<=90)]: plt.axvline(z, color='C2', ls='--', lw=0.6, alpha=0.45)
for z in nullE2[(nullE2>=0)&(nullE2<=90)]: plt.axvline(z, color='C3', ls='--', lw=0.6, alpha=0.45)
plt.scatter(thSL_H1[(thSL_H1>=0)&(thSL_H1<=90)], SL_H1[(thSL_H1>=0)&(thSL_H1<=90)], c='C0', s=18, zorder=3, label="Бічні H, 1-стр.")
plt.scatter(thSL_E1[(thSL_E1>=0)&(thSL_E1<=90)], SL_E1[(thSL_E1>=0)&(thSL_E1<=90)], c='C1', s=18, zorder=3, label="Бічні E, 1-стр.")
plt.scatter(thSL_H2[(thSL_H2>=0)&(thSL_H2<=90)], SL_H2[(thSL_H2>=0)&(thSL_H2<=90)], c='C2', s=18, zorder=3, label="Бічні H, 2-стр.")
plt.scatter(thSL_E2[(thSL_E2>=0)&(thSL_E2<=90)], SL_E2[(thSL_E2>=0)&(thSL_E2<=90)], c='C3', s=18, zorder=3, label="Бічні E, 2-стр.")
plt.title("ДС стрижневої антени: 1-стриж. vs 2-стриж. (лінійний масштаб)")
plt.xlabel("θ, град"); plt.ylabel("Амплітуда (норма на максимум)")
plt.xlim(0,90); plt.ylim(0,1.05); plt.grid(ls='--', alpha=0.3); plt.legend(loc="upper right")

plt.figure(figsize=(9,5.1))
plt.plot(deg, 20*np.log10(Fh1+1e-12), label="H, 1-стриж.")
plt.plot(deg, 20*np.log10(Fe1+1e-12), '--', label="E, 1-стриж.")
plt.plot(deg, 20*np.log10(Fh2+1e-12), label="H, 2-стриж.")
plt.plot(deg, 20*np.log10(Fe2+1e-12), '--', label="E, 2-стриж.")
plt.scatter(thSL_H1[(thSL_H1>=0)&(thSL_H1<=90)], SLdB_H1[(thSL_H1>=0)&(thSL_H1<=90)], c='C0', s=22, zorder=3, label="Бічні H, 1-стр. (дБ)")
plt.scatter(thSL_E1[(thSL_E1>=0)&(thSL_E1<=90)], SLdB_E1[(thSL_E1>=0)&(thSL_E1<=90)], c='C1', s=22, zorder=3, label="Бічні E, 1-стр. (дБ)")
plt.scatter(thSL_H2[(thSL_H2>=0)&(thSL_H2<=90)], SLdB_H2[(thSL_H2>=0)&(thSL_H2<=90)], c='C2', s=22, zorder=3, label="Бічні H, 2-стр. (дБ)")
plt.scatter(thSL_E2[(thSL_E2>=0)&(thSL_E2<=90)], SLdB_E2[(thSL_E2>=0)&(thSL_E2<=90)], c='C3', s=22, zorder=3, label="Бічні E, 2-стр. (дБ)")
plt.title("ДС стрижневої антени: 1-стриж. vs 2-стриж. (дБ)")
plt.xlabel("θ, град"); plt.ylabel("Рівень, дБ (норма на максимум)")
plt.xlim(0,90); plt.ylim(-60,0); plt.grid(ls='--', alpha=0.3); plt.legend(loc="upper right")

print(f"λ={lam:.3f} м (λ={lambda_cm} см), h={h:.3f} м (h={h_cm} см), ξ={xi:.3f}, d_eff={d_eff*100:.1f} см, L={L*100:.1f} см")
print(f"1-стриж.: HPBW_E≈{HP_E1:.2f}°, HPBW_H≈{HP_H1:.2f}°")
print(f"2-стриж.: HPBW_E≈{HP_E2:.2f}°, HPBW_H≈{HP_H2:.2f}°")

plt.show()
