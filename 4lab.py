import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.special import j1  # Bessel J1

# ---------- Вихідні дані: Варіант 2 ----------
lambda_cm = 3.2
h_cm      = 8.0
eps_r     = 2.2
tg_delta  = 2e-4
d_max_cm, d_min_cm, d_avg_cm = 2.9, 1.7, 2.3
L_cm = 23.7

# довжини у СІ
lam = lambda_cm/100.0
h   = h_cm/100.0
d_max = d_max_cm/100.0
d_min = d_min_cm/100.0
d_cp  = d_avg_cm/100.0
L     = L_cm/100.0
k = 2*np.pi/lam

# ---------- Коефіцієнт уповільнення xi (взяти з графіка п.5!) ----------
# Підстав СВОЄ значення з графіка (приклад: 1.10). Можна змінити тут.
xi = 1.10

# ---------- Лямбда-функція першого порядку Λ1(x) = 2*J1(x)/x ----------
def Lambda1(x):
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = 2.0*j1(x[nz])/x[nz]
    return out

# ---------- Частини формул методички ----------
# (4.3) F_b(θ) для решітки з хвилею, що біжить
def F_b(theta):
    num1 = (xi - 1.0) / np.sin(np.pi*L/lam * (xi - 1.0))
    term = np.sin(np.pi*L/lam * (xi - np.cos(theta))) / (xi - np.cos(theta))
    return np.abs(num1 * term)

# (4.4) F_1E(θ) = Λ1( (π d/λ) sinθ ) * cosθ
def F_1E(theta, d=d_cp):
    x = np.pi*d/lam * np.sin(theta)
    return np.abs(Lambda1(x) * np.cos(theta))

# (4.5) F_1H(θ) = Λ1( (π d/λ) sinθ )
def F_1H(theta, d=d_cp):
    x = np.pi*d/lam * np.sin(theta)
    return np.abs(Lambda1(x))

# (4.2) Повні ДС одно-стрижневої антени
def FE_one(theta):
    return F_b(theta) * F_1E(theta)

def FH_one(theta):
    return F_b(theta) * F_1H(theta)

# (4.16) ДС дво-стрижневої антени: множимо на F_C(θ) — інтерференційний множник
# Для вертикальної поляризації (збуджені стрижні вертикальні)
# приймаємо F_C(θ) = |2 cos( (k h/2) sinθ )|
def Fc(theta):
    return np.abs(2.0*np.cos(0.5*k*h*np.sin(theta)))

def FE_two(theta):
    return FE_one(theta) * Fc(theta)

def FH_two(theta):
    return FH_one(theta) * Fc(theta)

# ---------- Утиліти для HPBW/нулів/бічних ----------
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

# ---------- Сітки кутів ----------
deg_full = np.linspace(-90.0, 90.0, 18001)
th_full  = np.deg2rad(deg_full)

deg = np.linspace(0.0, 90.0, 9001)
th  = np.deg2rad(deg)

# ---------- Обчислення: обери 'one' або 'two' ----------
mode = 'two'  # 'one' — одно-стрижнева, 'two' — дво-стрижнева

def patterns(mode, theta):
    if mode == 'one':
        Fe = FE_one(theta); Fh = FH_one(theta)
    else:
        Fe = FE_two(theta); Fh = FH_two(theta)
    Fe = Fe/Fe.max() if Fe.max()!=0 else Fe
    Fh = Fh/Fh.max() if Fh.max()!=0 else Fh
    return Fe, Fh

Fe_full, Fh_full = patterns(mode, th_full)
Fe_disp, Fh_disp = patterns(mode, th)

# Метрики
HP_E, t1E, t2E = hpbw_deg(deg_full, Fe_full)
HP_H, t1H, t2H = hpbw_deg(deg_full, Fh_full)
nullE_full = nulls_deg(deg_full, Fe_full)
nullH_full = nulls_deg(deg_full, Fh_full)
slE = first_sidelobe(deg_full, Fe_full, t2E if np.isfinite(t2E) else 0)
slH = first_sidelobe(deg_full, Fh_full, t2H if np.isfinite(t2H) else 0)

nullE = nullE_full[(nullE_full >= 0) & (nullE_full <= 90)]
nullH = nullH_full[(nullH_full >= 0) & (nullH_full <= 90)]

# ---------- Графік (лінійний) ----------
plt.figure(figsize=(7.6, 4.4))
plt.plot(deg, Fh_disp, label=f"H-площина ({'2-стриж.' if mode=='two' else '1-стриж.'})")
plt.plot(deg, Fe_disp, '--', label=f"E-площина ({'2-стриж.' if mode=='two' else '1-стриж.'})")
plt.axhline(0.707, color='gray', ls='--', lw=0.8)
if np.isfinite(t2H): plt.axvline(t2H, color='C0', ls=':', lw=0.9)
if np.isfinite(t2E): plt.axvline(t2E, color='C1', ls=':', lw=0.9)
if len(nullH): plt.scatter(nullH, np.zeros_like(nullH), c='C0', s=18, zorder=3, label="Нулі H")
if len(nullE): plt.scatter(nullE, np.zeros_like(nullE), c='C1', s=18, zorder=3, label="Нулі E")
if slH and 0 <= slH[0] <= 90: plt.scatter([slH[0]], [slH[1]], c='C0', s=30, zorder=3, label="Бічна H")
if slE and 0 <= slE[0] <= 90: plt.scatter([slE[0]], [slE[1]], c='C1', s=30, zorder=3, label="Бічна E")
plt.title("Нормовані ДС діелектричної стрижневої антени (за методичкою)")
plt.xlabel("θ, град"); plt.ylabel("Амплітуда (норма на максимум)")
plt.xlim(0, 90); plt.ylim(0, 1.05); plt.grid(ls='--', alpha=0.3); plt.legend(loc="upper right")

# ---------- Графік (дБ) ----------
plt.figure(figsize=(7.6, 4.4))
plt.plot(deg, 20*np.log10(Fh_disp+1e-12), label="H-площина")
plt.plot(deg, 20*np.log10(Fe_disp+1e-12), '--', label="E-площина")
if slH and 0 <= slH[0] <= 90: plt.scatter([slH[0]], [20*np.log10(slH[1]+1e-12)], c='C0', s=30, zorder=3, label="1-ша бічна H")
if slE and 0 <= slE[0] <= 90: plt.scatter([slE[0]], [20*np.log10(slE[1]+1e-12)], c='C1', s=30, zorder=3, label="1-ша бічна E")
plt.title("Нормовані ДС діелектричної стрижневої антени (дБ)")
plt.xlabel("θ, град"); plt.ylabel("Рівень, дБ (норма на максимум)")
plt.xlim(0, 90); plt.ylim(-60, 0); plt.grid(ls='--', alpha=0.3); plt.legend(loc="upper right")

# ---------- Числовий підсумок ----------
print(f"λ = {lam:.3f} м, h = {h:.3f} м (λ={lambda_cm} см, h={h_cm} см), ξ = {xi:.3f}, d_cp = {d_cp:.3f} м")
print(f"HPBW_E ≈ {HP_E:.2f}°  (межі {t1E:.2f}° .. {t2E:.2f}°)")
print(f"HPBW_H ≈ {HP_H:.2f}°  (межі {t1H:.2f}° .. {t2H:.2f}°)")
if slE: print(f"1-ша бічна (E): ампл={slE[1]:.3f}, рівень={20*np.log10(slE[1]+1e-12):.2f} дБ @ θ≈{slE[0]:.2f}°")
if slH: print(f"1-ша бічна (H): ампл={slH[1]:.3f}, рівень={20*np.log10(slH[1]+1e-12):.2f} дБ @ θ≈{slH[0]:.2f}°")

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.special import j1  # Bessel J1

# ---------- Вихідні дані: Варіант 2 ----------
lambda_cm = 3.2
h_cm      = 8.0
eps_r     = 2.2
tg_delta  = 2e-4
d_max_cm, d_min_cm, d_avg_cm = 2.9, 1.7, 2.3
L_cm = 23.7

# довжини у СІ
lam = lambda_cm/100.0
h   = h_cm/100.0
d_max = d_max_cm/100.0
d_min = d_min_cm/100.0
d_cp  = d_avg_cm/100.0
L     = L_cm/100.0
k = 2*np.pi/lam

# ---------- Коефіцієнт уповільнення xi (взяти з графіка п.5!) ----------
# Підстав СВОЄ значення з графіка (приклад: 1.10). Можна змінити тут.
xi = 1.10

# ---------- Лямбда-функція першого порядку Λ1(x) = 2*J1(x)/x ----------
def Lambda1(x):
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = 2.0*j1(x[nz])/x[nz]
    return out

# ---------- Частини формул методички ----------
# (4.3) F_b(θ) для решітки з хвилею, що біжить
def F_b(theta):
    num1 = (xi - 1.0) / np.sin(np.pi*L/lam * (xi - 1.0))
    term = np.sin(np.pi*L/lam * (xi - np.cos(theta))) / (xi - np.cos(theta))
    return np.abs(num1 * term)

# (4.4) F_1E(θ) = Λ1( (π d/λ) sinθ ) * cosθ
def F_1E(theta, d=d_cp):
    x = np.pi*d/lam * np.sin(theta)
    return np.abs(Lambda1(x) * np.cos(theta))

# (4.5) F_1H(θ) = Λ1( (π d/λ) sinθ )
def F_1H(theta, d=d_cp):
    x = np.pi*d/lam * np.sin(theta)
    return np.abs(Lambda1(x))

# (4.2) Повні ДС одно-стрижневої антени
def FE_one(theta):
    return F_b(theta) * F_1E(theta)

def FH_one(theta):
    return F_b(theta) * F_1H(theta)

# (4.16) ДС дво-стрижневої антени: множимо на F_C(θ) — інтерференційний множник
# Для вертикальної поляризації (збуджені стрижні вертикальні)
# приймаємо F_C(θ) = |2 cos( (k h/2) sinθ )|
def Fc(theta):
    return np.abs(2.0*np.cos(0.5*k*h*np.sin(theta)))

def FE_two(theta):
    return FE_one(theta) * Fc(theta)

def FH_two(theta):
    return FH_one(theta) * Fc(theta)

# ---------- Утиліти для HPBW/нулів/бічних ----------
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

# ---------- Сітки кутів ----------
deg_full = np.linspace(-90.0, 90.0, 18001)
th_full  = np.deg2rad(deg_full)

deg = np.linspace(0.0, 90.0, 9001)
th  = np.deg2rad(deg)

# ---------- Обчислення: обери 'one' або 'two' ----------
mode = 'two'  # 'one' — одно-стрижнева, 'two' — дво-стрижнева

def patterns(mode, theta):
    if mode == 'one':
        Fe = FE_one(theta); Fh = FH_one(theta)
    else:
        Fe = FE_two(theta); Fh = FH_two(theta)
    Fe = Fe/Fe.max() if Fe.max()!=0 else Fe
    Fh = Fh/Fh.max() if Fh.max()!=0 else Fh
    return Fe, Fh

Fe_full, Fh_full = patterns(mode, th_full)
Fe_disp, Fh_disp = patterns(mode, th)

# Метрики
HP_E, t1E, t2E = hpbw_deg(deg_full, Fe_full)
HP_H, t1H, t2H = hpbw_deg(deg_full, Fh_full)
nullE_full = nulls_deg(deg_full, Fe_full)
nullH_full = nulls_deg(deg_full, Fh_full)
slE = first_sidelobe(deg_full, Fe_full, t2E if np.isfinite(t2E) else 0)
slH = first_sidelobe(deg_full, Fh_full, t2H if np.isfinite(t2H) else 0)

nullE = nullE_full[(nullE_full >= 0) & (nullE_full <= 90)]
nullH = nullH_full[(nullH_full >= 0) & (nullH_full <= 90)]

# ---------- Графік (лінійний) ----------
plt.figure(figsize=(7.6, 4.4))
plt.plot(deg, Fh_disp, label=f"H-площина ({'2-стриж.' if mode=='two' else '1-стриж.'})")
plt.plot(deg, Fe_disp, '--', label=f"E-площина ({'2-стриж.' if mode=='two' else '1-стриж.'})")
plt.axhline(0.707, color='gray', ls='--', lw=0.8)
if np.isfinite(t2H): plt.axvline(t2H, color='C0', ls=':', lw=0.9)
if np.isfinite(t2E): plt.axvline(t2E, color='C1', ls=':', lw=0.9)
if len(nullH): plt.scatter(nullH, np.zeros_like(nullH), c='C0', s=18, zorder=3, label="Нулі H")
if len(nullE): plt.scatter(nullE, np.zeros_like(nullE), c='C1', s=18, zorder=3, label="Нулі E")
if slH and 0 <= slH[0] <= 90: plt.scatter([slH[0]], [slH[1]], c='C0', s=30, zorder=3, label="Бічна H")
if slE and 0 <= slE[0] <= 90: plt.scatter([slE[0]], [slE[1]], c='C1', s=30, zorder=3, label="Бічна E")
plt.title("Нормовані ДС діелектричної стрижневої антени (за методичкою)")
plt.xlabel("θ, град"); plt.ylabel("Амплітуда (норма на максимум)")
plt.xlim(0, 90); plt.ylim(0, 1.05); plt.grid(ls='--', alpha=0.3); plt.legend(loc="upper right")

# ---------- Графік (дБ) ----------
plt.figure(figsize=(7.6, 4.4))
plt.plot(deg, 20*np.log10(Fh_disp+1e-12), label="H-площина")
plt.plot(deg, 20*np.log10(Fe_disp+1e-12), '--', label="E-площина")
if slH and 0 <= slH[0] <= 90: plt.scatter([slH[0]], [20*np.log10(slH[1]+1e-12)], c='C0', s=30, zorder=3, label="1-ша бічна H")
if slE and 0 <= slE[0] <= 90: plt.scatter([slE[0]], [20*np.log10(slE[1]+1e-12)], c='C1', s=30, zorder=3, label="1-ша бічна E")
plt.title("Нормовані ДС діелектричної стрижневої антени (дБ)")
plt.xlabel("θ, град"); plt.ylabel("Рівень, дБ (норма на максимум)")
plt.xlim(0, 90); plt.ylim(-60, 0); plt.grid(ls='--', alpha=0.3); plt.legend(loc="upper right")

# ---------- Числовий підсумок ----------
print(f"λ = {lam:.3f} м, h = {h:.3f} м (λ={lambda_cm} см, h={h_cm} см), ξ = {xi:.3f}, d_cp = {d_cp:.3f} м")
print(f"HPBW_E ≈ {HP_E:.2f}°  (межі {t1E:.2f}° .. {t2E:.2f}°)")
print(f"HPBW_H ≈ {HP_H:.2f}°  (межі {t1H:.2f}° .. {t2H:.2f}°)")
if slE: print(f"1-ша бічна (E): ампл={slE[1]:.3f}, рівень={20*np.log10(slE[1]+1e-12):.2f} дБ @ θ≈{slE[0]:.2f}°")
if slH: print(f"1-ша бічна (H): ампл={slH[1]:.3f}, рівень={20*np.log10(slH[1]+1e-12):.2f} дБ @ θ≈{slH[0]:.2f}°")

plt.show()
