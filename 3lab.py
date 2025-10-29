import numpy as np
import math
import matplotlib.pyplot as plt

N = 7
f = 420e6
c = 299_792_458.0
lam = c / f
d = 0.25 * lam
k = 2*np.pi / lam
beta = -k*d

use_dipole_element = True

def sinc_safe(x):
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-15
    out[nz] = np.sin(x[nz]) / x[nz]
    return out

def array_factor(theta):
    psi = k*d*np.cos(theta) + beta
    return np.sin(N*psi/2.0) / (N * np.sin(psi/2.0) + 1e-15)

def dipole_elem_factor(theta):
    x = (np.pi/2.0)*np.cos(theta)
    num = np.cos(x)
    den = np.sin(theta) + 1e-15
    return np.abs(num/den)

def pattern(theta):
    AF = np.abs(array_factor(theta))
    Fe = dipole_elem_factor(theta) if use_dipole_element else 1.0
    P = AF * Fe
    return P / np.max(P)

def to_dB(x):
    x = np.maximum(x, 1e-12)
    return 20*np.log10(x)

def half_power_bw_deg(angles_deg, pat):
    peak_i = np.argmax(pat)
    peak = pat[peak_i]
    target = 0.707*peak

    i_left = peak_i
    while i_left > 0 and pat[i_left] > target:
        i_left -= 1
    x1, y1 = angles_deg[i_left], pat[i_left]
    x2, y2 = angles_deg[i_left+1], pat[i_left+1]
    if y2 != y1:
        left_deg = x1 + (target - y1) * (x2 - x1) / (y2 - y1)
    else:
        left_deg = angles_deg[i_left]

    i_right = peak_i
    while i_right < len(pat)-1 and pat[i_right] > target:
        i_right += 1
    x1, y1 = angles_deg[i_right-1], pat[i_right-1]
    x2, y2 = angles_deg[i_right],   pat[i_right]
    if y2 != y1:
        right_deg = x1 + (target - y1) * (x2 - x1) / (y2 - y1)
    else:
        right_deg = angles_deg[i_right]

    return (right_deg - left_deg), left_deg, right_deg

def first_sidelobe_level_dB(pat_dB):
    p = pat_dB.copy()
    i0 = np.argmax(p)
    width = 50
    l = max(0, i0 - width)
    r = min(len(p), i0 + width)
    p[l:r] = -1e9
    idx = np.argmax(p)
    return float(p[idx])

angles_deg = np.linspace(0, 90, 40001)
theta = np.deg2rad(angles_deg)

P_E = pattern(theta)
P_H = pattern(theta)

bwE, lE, rE = half_power_bw_deg(angles_deg, P_E)
bwH, lH, rH = half_power_bw_deg(angles_deg, P_H)

P_E_dB = to_dB(P_E)
P_H_dB = to_dB(P_H)
sllE = first_sidelobe_level_dB(P_E_dB)
sllH = first_sidelobe_level_dB(P_H_dB)

print(f"λ = {lam:.4f} м,  d = {d:.4f} м ({d/lam:.3f} λ),  k = {k:.4f} рад/м")
print(f"N = {N}, β = {beta:.4f} рад (ендфаєр)")

print("\nШирина головної пелюстки (на рівні 0.707, тобто −3 dB):")
print(f"  E-площина: {bwE:.2f}°  (ліва межа {lE:.2f}°, права {rE:.2f}°)")
print(f"  H-площина: {bwH:.2f}°  (ліва межа {lH:.2f}°, права {rH:.2f}°)")

print("\nРівень найбільшої бічної пелюстки (SLL, dB):")
print(f"  E-площина: {sllE:.2f} dB")
print(f"  H-площина: {sllH:.2f} dB")


plt.figure(figsize=(9,5))
plt.plot(angles_deg, P_H, label="H-площина")
plt.plot(angles_deg, P_E, "--", label="E-площина")
plt.axhline(0.707, lw=0.8, color="gray")
plt.axvline(lE, lw=0.6, color="gray", ls="--")
plt.axvline(rE, lw=0.6, color="gray", ls="--")
plt.title("Нормовані ДС директорної антени (лінійний масштаб)")
plt.xlabel("θ, град")
plt.ylabel("Амплітуда (норма на максимум)")
plt.xlim(0, 90)
plt.ylim(0, 1.05)
plt.grid(True, ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("patterns_linear.png", dpi=300)

# В dB
plt.figure(figsize=(9,5))
plt.plot(angles_deg, P_H_dB, label="H-площина")
plt.plot(angles_deg, P_E_dB, "--", label="E-площина")
plt.axhline(-3, lw=0.8, color="gray")
plt.title("Нормовані ДС директорної антени (dB)")
plt.xlabel("θ, град")
plt.ylabel("Рівень, dB (норма на максимум)")
plt.xlim(0, 90)
plt.ylim(-50, 0)
plt.grid(True, ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("patterns_dB.png", dpi=300)

plt.show()
