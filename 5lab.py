import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ===== Вихідні (варіант 2, тип 2) =====
lambda0_cm = 3.3
lambda_g_cm = 4.8
N  = 5
Ls_cm = 2.0
d_cm  = 4.0
psi0  = 0.0

# ===== Перехід у СІ =====
lam0 = lambda0_cm/100.0
lamg = lambda_g_cm/100.0
k0   = 2*np.pi/lam0
beta_g = 2*np.pi/lamg
Ls = Ls_cm/100.0
d  = d_cm/100.0

def sinc(x):
    y = np.ones_like(x)
    m = x != 0
    y[m] = np.sin(x[m])/x[m]
    return y

# Елементна ДС щілини (зелена)
def slot_H(theta):
    u = 0.5*k0*Ls*np.sin(theta)
    return np.abs(sinc(u)*np.cos(theta))

# Масивний фактор (синя) із хвилеводною фазою
def AF(theta):
    psi = beta_g*d - k0*d*np.sin(theta) + psi0
    num = np.sin(N*psi/2.0)
    den = np.sin(psi/2.0)
    y = np.ones_like(theta, dtype=float)
    m = np.abs(den) > 1e-12
    y[m] = np.abs(num[m]/den[m])
    y[~m] = N
    return y/N

# Повна ДС у H-площині (помаранчева)
def FH(theta):
    return slot_H(theta)*AF(theta)

# ===== Сервісні =====
def interp_x_at_y(x, y, y0):
    i = np.where((y[:-1]-y0)*(y[1:]-y0) <= 0)[0]
    xs=[]
    for k in i:
        x1,x2=x[k],x[k+1]; y1,y2=y[k],y[k+1]
        xs.append(x1 if y2==y1 else x1 + (y0-y1)*(x2-x1)/(y2-y1))
    return np.array(xs)

def all_sidelobes(x_deg, F):
    idx,_ = find_peaks(F)
    return x_deg[idx], F[idx], 20*np.log10(F[idx]+1e-12)

def hpbw_of_main_lobe(x_deg, F):
    xs = np.deg2rad(x_deg)
    lvl = 0.707
    idx = np.where((F[:-1]-lvl)*(F[1:]-lvl) <= 0)[0]
    if len(idx) == 0:
        return np.nan, np.nan, np.nan
    th = []
    for i in idx:
        x1,x2 = xs[i], xs[i+1]; y1,y2 = F[i], F[i+1]
        th.append(x1 if y2==y1 else x1 + (lvl-y1)*(x2-x1)/(y2-y1))
    th = np.array(th)
    lefts  = th[th < 0]
    rights = th[th > 0]
    if len(lefts)==0 or len(rights)==0:
        return np.nan, np.nan, np.nan
    t1 = lefts.max(); t2 = rights.min()
    return np.rad2deg(t2 - t1), np.rad2deg(t1), np.rad2deg(t2)

# ===== Аналітичні НУЛІ =====
def analytic_nulls_deg():
    full = set()

    # 1) Нулі елементної ДС: cos(theta)=0 -> ±90°
    full.update([-90.0, 90.0])

    #    та нулі sinc: (k0*Ls/2)*sinθ = n*pi -> sinθ = 2n*pi/(k0*Ls)
    n_max = int(np.floor((k0*Ls)/(2*np.pi)))
    for n in range(1, n_max+1):
        sval = 2*np.pi*n/(k0*Ls)
        if abs(sval) <= 1:
            t = np.degrees(np.arcsin(sval))
            for cand in [t, 180.0-t, -t, -180.0+t]:
                if -90.0 <= cand <= 90.0:
                    full.add(float(np.round(cand, 6)))

    # 2) Нулі AF: sin(N*psi/2)=0 (крім полюсів sin(psi/2)=0)
    m_range = range(-3*N, 3*N+1)
    for m in m_range:
        target = 2*np.pi*m / N
        sval = (beta_g*d - target)/(k0*d)  # sinθ
        if abs(sval) <= 1:
            theta_deg = float(np.degrees(np.arcsin(sval)))
            psi_val = beta_g*d - k0*d*np.sin(np.radians(theta_deg))
            if not np.isclose(np.sin(psi_val/2), 0.0, atol=1e-9):
                if -90.0 <= theta_deg <= 90.0:
                    full.add(float(np.round(theta_deg, 6)))

    return sorted(full)

# ===== Сітка кутів =====
deg = np.linspace(-90.0, 90.0, 18001)
th  = np.deg2rad(deg)

# ===== Криві =====
AF_lin   = AF(th)                 # синя
SLOT_lin = slot_H(th)             # зелена
FH_lin   = FH(th); FH_lin /= FH_lin.max()  # помаранчева (нормована)
AF_lin   = AF_lin / AF_lin.max()
SLOT_lin = SLOT_lin / SLOT_lin.max()

# Аналітичні нулі, бічні, HPBW
full_zeros_deg = analytic_nulls_deg()
thSL, SLamp, SLdB = all_sidelobes(deg, FH_lin)
HPBW, th_left, th_right = hpbw_of_main_lobe(deg, FH_lin)

# ===== Графік (лінійний) =====
plt.figure(figsize=(9,5.2))
plt.plot(deg, AF_lin,   color='tab:blue',   label='|AF(θ)| — масивний фактор')
plt.plot(deg, SLOT_lin, color='tab:green',  label='|F_slot(θ)| — елементна ДС')
plt.plot(deg, FH_lin,   color='tab:orange', label='H-площина (повна ДС)')
# аналітичні НУЛІ
for z in full_zeros_deg:
    plt.axvline(z, color='tab:orange', ls='--', lw=0.8, alpha=0.6)
# бічні
plt.scatter(thSL, SLamp, c='tab:blue', s=20, zorder=3, label='бічні пелюстки')
# рівень 0.707 і межі HPBW
plt.axhline(0.707, color='gray', ls='--', lw=0.9, label='рівень 0.707')
if not np.isnan(HPBW):
    plt.axvline(th_left,  color='gray', ls=':', lw=0.9)
    plt.axvline(th_right, color='gray', ls=':', lw=0.9)
    plt.text((th_left+th_right)/2, 0.73, f'HPBW ≈ {HPBW:.2f}°',
             ha='center', va='bottom', fontsize=9)

plt.title('Нормована ДС ХвЩА (тип 2) — H-площина')
plt.xlabel('θ, град'); plt.ylabel('Амплітуда (норма на максимум)')
plt.xlim(-90, 90); plt.ylim(0, 1.05); plt.grid(ls='--', alpha=0.3); plt.legend(loc='upper right')

# ===== Графік (дБ) =====
plt.figure(figsize=(9,5.2))
plt.plot(deg, 20*np.log10(FH_lin+1e-12), color='tab:orange', label='H-площина (повна ДС)')
plt.scatter(thSL, SLdB, c='tab:blue', s=20, zorder=3, label='бічні (дБ)')
for z in full_zeros_deg:
    plt.axvline(z, color='tab:orange', ls='--', lw=0.8, alpha=0.6)
plt.title('Нормована ДС ХвЩА (тип 2), H-площина (дБ)')
plt.xlabel('θ, град'); plt.ylabel('Рівень, дБ (норма на максимум)')
plt.xlim(-90, 90); plt.ylim(-60, 0); plt.grid(ls='--', alpha=0.3); plt.legend(loc='upper right')

# ===== Консольний підсумок =====
print(f"λ0={lam0:.3f} м (λ0={lambda0_cm} см), λg={lamg:.3f} м (λg={lambda_g_cm} см)")
print(f"N={N}, Ls={Ls_cm} см, d={d_cm} см, ψ0={psi0:.2f} рад")
print("Нулі (аналітичні, град):", ", ".join(f"{z:.2f}" for z in full_zeros_deg))
if not np.isnan(HPBW):
    print(f"HPBW_H ≈ {HPBW:.2f}°  (межі {th_left:.2f}° .. {th_right:.2f}°)")
else:
    print("HPBW_H: не знайдено перетинів із 0.707")

plt.show()
