import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.special import j0, j1, jn, jn_zeros  # jn(2,x)=J2(x)

# ================== Вхідні дані (приклад з рисунка викладача) ==================
lambda_cm = 3.2      # довжина хвилі, см
D_m       = 0.9      # діаметр дзеркала, м
f_m       = 0.4      # фокусна відстань, м

# ================== Базові величини з методички ==================
lam = lambda_cm / 100.0
k   = 2*np.pi / lam            # хвильове число
R0  = D_m / 2.0                # радіус дзеркала
p   = 2.0 * f_m                # подвоєний фокус
v   = 3.5 * R0 / p             # параметр v
eps = 1e-12

def u_of_theta(theta):         # theta в радіанах
    return k * R0 * np.sin(theta)

def J1_over_x(x):              # безпечний J1(x)/x, lim_{x→0} = 1/2
    x = np.asarray(x)
    out = np.empty_like(x, dtype=float)
    m = np.abs(x) > 1e-10
    out[m] = j1(x[m]) / x[m]
    out[~m] = 0.5
    return out

def safe_div(a, b):            # безпечне ділення (уникнути 0 в знаменнику)
    return a / (b + np.where(b >= 0, eps, -eps))

den_norm = 0.74 * J1_over_x(v) + 0.13   # нормувальний дільник (скаляр)

# спільні доданки у фігурних дужках в FH/FE
def core_terms(u):
    u = np.asarray(u)
    # t1 = 0.74 * ( v J1(v) J0(u) − u J1(u) J0(v) ) / (v^2 − u^2)
    num1 = v*j1(v)*j0(u) - u*j1(u)*j0(v)
    den1 = v**2 - u**2
    t1 = 0.74 * safe_div(num1, den1)
    t1 = np.where(np.abs(u) < 1e-10, 0.74*(j1(v)/v), t1)    # ліміт при u→0

    # t2 = 0.26 * J1(u)/u
    t2 = 0.26 * J1_over_x(u)

    # t3 = ( u J1(u) J2(1.5v) − 1.5 v J1(1.5v) J2(u) ) / ( (1.5v)^2 − u^2 )
    num3 = u*j1(u)*jn(2, 1.5*v) - 1.5*v*j1(1.5*v)*jn(2, u)
    den3 = (1.5*v)**2 - u**2
    t3 = safe_div(num3, den3)
    t3 = np.where(np.abs(u) < 1e-10, 0.0, t3)              # ліміт при u→0
    return t1, t2, t3

def FH(theta):
    u = u_of_theta(theta)
    t1, t2, t3 = core_terms(u)
    inner = t1 + t2 - 0.25*t3
    return (np.cos(theta/2.0)**2) * safe_div(inner, den_norm)

def FE(theta):
    u = u_of_theta(theta)
    t1, t2, t3 = core_terms(u)
    inner = t1 + t2 + 0.25*t3
    return (np.cos(theta/2.0)**2) * safe_div(inner, den_norm)

# ================== Сітки ==================
deg_full = np.linspace(-90.0, 90.0, 36001)
th_full  = np.deg2rad(deg_full)
deg_disp = np.linspace(0.0, 90.0, 18001)
th_disp  = np.deg2rad(deg_disp)
x_full   = k * R0 * np.sin(th_full)
x_disp   = k * R0 * np.sin(th_disp)

# ================== Поля (зі знаком) + нормування ==================
FH_full = np.nan_to_num(FH(th_full))
FE_full = np.nan_to_num(FE(th_full))
norm = max(np.max(np.abs(FH_full)), np.max(np.abs(FE_full)), 1e-12)
FH_full_n = FH_full / norm
FE_full_n = FE_full / norm
FH_disp_n = np.nan_to_num(FH(th_disp) / norm)
FE_disp_n = np.nan_to_num(FE(th_disp) / norm)

# ================== Нулі (зміна знака на повній сітці) ==================
def zeros_deg(x_deg, F):
    xs = np.deg2rad(x_deg)
    idx = np.where((F[:-1] * F[1:]) < 0)[0]
    thz = []
    for i in idx:
        x1, x2 = xs[i], xs[i+1]
        y1, y2 = F[i], F[i+1]
        thz.append(x1 if abs(y2 - y1) < 1e-15 else x1 + (0 - y1) * (x2 - x1) / (y2 - y1))
    return np.rad2deg(np.array(thz)) if len(thz) else np.array([])

zeros_E = zeros_deg(deg_full, FE_full_n)
zeros_H = zeros_deg(deg_full, FH_full_n)

# ================== HPBW (|F|=sqrt(0.5) на повній сітці) ==================
def hpbw_deg(x_deg, F):
    xs = np.deg2rad(x_deg)
    A  = np.abs(F)
    lvl = np.sqrt(0.5)
    idx = np.where((A[:-1]-lvl)*(A[1:]-lvl) <= 0)[0]
    if len(idx)==0: return np.nan, np.nan, np.nan
    th = []
    for i in idx:
        x1,x2 = xs[i], xs[i+1]; y1,y2 = A[i], A[i+1]
        th.append(x1 if abs(y2-y1)<1e-15 else x1 + (lvl-y1)*(x2-x1)/(y2-y1))
    th = np.array(th)
    lefts  = th[th<0]; rights = th[th>0]
    if len(lefts)==0 or len(rights)==0: return np.nan, np.nan, np.nan
    t1 = lefts.max(); t2 = rights.min()
    return np.rad2deg(t2 - t1), np.rad2deg(t1), np.rad2deg(t2)

HP_E, t1E, t2E = hpbw_deg(deg_full, FE_full_n)
HP_H, t1H, t2H = hpbw_deg(deg_full, FH_full_n)

# ================== Бічні пелюстки (локальні max |F| поза головною) ==================
def sidelobes_all(x_deg, F, main_left, main_right):
    A = np.abs(F)
    mask = (x_deg<main_left)|(x_deg>main_right) if np.isfinite(main_left) and np.isfinite(main_right) else np.ones_like(A, bool)
    idx,_ = find_peaks(A[mask])
    th  = x_deg[mask][idx]
    val = A[mask][idx]
    val_db = 20*np.log10(val + 1e-12)
    return th, val, val_db

thSL_E, SL_E, SLdB_E = sidelobes_all(deg_full, FE_full_n, t1E, t2E)
thSL_H, SL_H, SLdB_H = sidelobes_all(deg_full, FH_full_n, t1H, t2H)

# ================== Графік 1: E(x)=F(θ) зі знаком ==================
plt.figure(figsize=(10,4.8))
plt.plot(x_disp, FH_disp_n, label='H-площина: F_H(x)', color='tab:blue')
plt.plot(x_disp, FE_disp_n, '--', label='E-площина: F_E(x)', color='tab:orange')

zeros_front_E = zeros_E[(zeros_E>=0)&(zeros_E<=90)]
zeros_front_H = zeros_H[(zeros_H>=0)&(zeros_H<=90)]
x_zeros_E = k*R0*np.sin(np.deg2rad(zeros_front_E))
x_zeros_H = k*R0*np.sin(np.deg2rad(zeros_front_H))
plt.scatter(x_zeros_H, np.zeros_like(x_zeros_H), c='tab:blue',  s=26, zorder=3, label='нулі H')
plt.scatter(x_zeros_E, np.zeros_like(x_zeros_E), c='tab:red',   s=26, zorder=3, label='нулі E')

idx_peaks_H,_ = find_peaks(np.abs(FH_disp_n))
idx_peaks_E,_ = find_peaks(np.abs(FE_disp_n))
plt.scatter(x_disp[idx_peaks_H], FH_disp_n[idx_peaks_H], c='0.2', s=20, zorder=3, label='бічні H (ампл.)')
plt.scatter(x_disp[idx_peaks_E], FE_disp_n[idx_peaks_E], c='0.5', s=20, zorder=3, label='бічні E (ампл.)')

plt.title(f"ДС ДЗА у координатах x = k R0 sin(θ) (поле зі знаком)\nλ={lam:.3f} м, D={D_m:.2f} м, f={f_m:.2f} м")
plt.xlabel("x = k R0 sin(θ)")
plt.ylabel("F(θ) / max|F|")
plt.xlim(0, x_disp.max()); plt.grid(ls='--', alpha=0.3); plt.legend(loc='upper right')

# ================== Графік 2: 20*log10(|F(θ)|) від θ ==================
plt.figure(figsize=(10,4.8))
front = (deg_full>=0)&(deg_full<=90)
AH = 20*np.log10(np.abs(FH_full_n[front]) + 1e-12)
AE = 20*np.log10(np.abs(FE_full_n[front]) + 1e-12)
plt.plot(deg_full[front], AH, label='H-площина')
plt.plot(deg_full[front], AE, '--', label='E-площина')

mH = (thSL_H>=0)&(thSL_H<=90)
mE = (thSL_E>=0)&(thSL_E<=90)
plt.scatter(thSL_H[mH], SLdB_H[mH], c='tab:blue',  s=24, zorder=3, label='бічні H (дБ)')
plt.scatter(thSL_E[mE], SLdB_E[mE], c='tab:orange',s=24, zorder=3, label='бічні E (дБ)')

# аналітичні нулі через x=α_n -> θ = arcsin(α_n/(kR0))
alpha = jn_zeros(1, 12)
theta_zeros_front = []
for an in alpha:
    s = an / (k * R0)
    if 0 <= s <= 1:
        theta_zeros_front.append(np.degrees(np.arcsin(s)))
for z in theta_zeros_front:
    plt.axvline(z, color='k', ls='--', lw=0.6, alpha=0.5)

if np.isfinite(t2H): plt.axvline(t2H, color='tab:blue',  ls=':', lw=0.9)
if np.isfinite(t2E): plt.axvline(t2E, color='tab:orange',ls=':', lw=0.9)

plt.title("Нормовані ДС дзеркальної антени — дБ (|F|)")
plt.xlabel("θ, град"); plt.ylabel("20·log10(|F| / max|F|)")
plt.xlim(0,90); plt.ylim(-60,0); plt.grid(ls='--', alpha=0.3); plt.legend(loc='upper right')

# ================== Вивід чисел ==================
print(f"lambda={lam:.3f} m (={lambda_cm} cm), D={D_m:.3f} m, f={f_m:.3f} m")
print(f"HPBW_E ≈ {HP_E:.3f}°  (θ1={t1E:.3f}° .. θ2={t2E:.3f}°)")
print(f"HPBW_H ≈ {HP_H:.3f}°  (θ1={t1H:.3f}° .. {t2H:.3f}°)")
print("Нулі E (0..90°):", ", ".join(f"{z:.2f}°" for z in zeros_front_E))
print("Нулі H (0..90°):", ", ".join(f"{z:.2f}°" for z in zeros_front_H))

plt.show()
