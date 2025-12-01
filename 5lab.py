import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Вихідні (варіант 2, тип 2)
lambda0_cm = 3.3
lambda_g_cm = 4.8      # постав свою λg за методичкою
N = 5
Ls_cm = 2.0
d_cm  = 4.0
psi0  = 0.0

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

# Елементна ДС (зелена)
def slot_H(theta):
    u = 0.5*k0*Ls*np.sin(theta)
    return np.abs(sinc(u)*np.cos(theta))

# Масивний фактор (синя)
def AF(theta):
    psi = beta_g*d - k0*d*np.sin(theta) + psi0
    num = np.sin(N*psi/2.0)
    den = np.sin(psi/2.0)
    y = np.ones_like(theta, dtype=float)
    m = np.abs(den) > 1e-12
    y[m] = np.abs(num[m]/den[m])
    y[~m] = N
    return y/N

# Повна ДС (помаранчева)
def FH(theta):
    return slot_H(theta)*AF(theta)

def interp_x_at_y(x, y, y0):
    i = np.where((y[:-1]-y0)*(y[1:]-y0) <= 0)[0]
    xs=[]
    for k in i:
        x1,x2=x[k],x[k+1]; y1,y2=y[k],y[k+1]
        xs.append(x1 if y2==y1 else x1 + (y0-y1)*(x2-x1)/(y2-y1))
    return np.array(xs)

def all_nulls_deg(x_deg, F):
    xs = np.deg2rad(x_deg)
    z = interp_x_at_y(xs, F, 0.0)
    return np.rad2deg(z[(z>=-np.pi/2)&(z<=np.pi/2)])

def all_sidelobes(x_deg, F):
    idx,_ = find_peaks(F)
    return x_deg[idx], F[idx], 20*np.log10(F[idx]+1e-12)

deg = np.linspace(-90.0, 90.0, 18001)
th  = np.deg2rad(deg)

# Обчислення
AF_lin   = AF(th)                 # синя
SLOT_lin = slot_H(th)             # зелена
FH_lin   = FH(th); FH_lin /= FH_lin.max()  # помаранчева (нормована)
AF_lin   = AF_lin / AF_lin.max()
SLOT_lin = SLOT_lin / SLOT_lin.max()

nulls = all_nulls_deg(deg, FH_lin)

# Графік (лінійний) з трьома підписаними кривими
plt.figure(figsize=(9,5.2))
plt.plot(deg, AF_lin,   color='tab:blue',   label='|AF(θ)| — масивний фактор')
plt.plot(deg, SLOT_lin, color='tab:green',  label='|F_slot(θ)| — елементна ДС')
plt.plot(deg, FH_lin,   color='tab:orange', label='H-площина (повна ДС)')
for z in nulls: plt.axvline(z, color='tab:orange', ls='--', lw=0.8, alpha=0.6)
plt.scatter(thSL, SLamp, c='tab:blue', s=20, zorder=3, label='бічні пелюстки')
plt.axhline(0.707, color='gray', ls='--', lw=0.8, label='рівень 0.707')
plt.title(f'Нормована ДС ХвЩА (тип 2)')
plt.xlabel('θ, град'); plt.ylabel('Амплітуда (норма на максимум)')
plt.xlim(-90, 90); plt.ylim(0, 1.05); plt.grid(ls='--', alpha=0.3); plt.legend(loc='upper right')

# ДБ-графік (за бажанням — залишаю)
plt.figure(figsize=(9,5.2))
plt.plot(deg, 20*np.log10(FH_lin+1e-12), color='tab:orange', label='H-площина (повна ДС)')
plt.scatter(thSL, SLdB, c='tab:blue', s=20, zorder=3, label='бічні (дБ)')
for z in nulls: plt.axvline(z, color='tab:orange', ls='--', lw=0.8, alpha=0.6)
plt.title('Нормована ДС ХвЩА (тип 2), H-площина (дБ)')
plt.xlabel('θ, град'); plt.ylabel('Рівень, дБ (норма на максимум)')
plt.xlim(-90, 90); plt.ylim(-60, 0); plt.grid(ls='--', alpha=0.3); plt.legend(loc='upper right')

plt.show()
