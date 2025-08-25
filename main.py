import sectionFunction as fn
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt

x, data = fn.getSections("wig_seakeeping.txt")
T = abs(data[0,:]).max() #Only works for Wigley hull
B_N = 2*abs(data[:,0]).max()
L = 120

B = []
for i in range(len(data)):
    B.append((2*fn.getMaxB(data[i,0]))/T)
    ratioB = np.array(B)
vol = fn.getVolume(data,x)
density = 1.025
M = round(vol*density, 4)
C33 = fn.getC33(data, x)

lamb = np.linspace(0.2, 2,500)
omega = np.zeros(len(lamb))
for i in range(len(lamb)):
    omega[i] = math.sqrt((2*3.14*9.81)/(lamb[i]*L))
#print(omega)
A_ = []
B_ = []
tuningFactor = []
k = []

for i in range(len(omega)):
    w_e, x_curve = fn.inputFunc(x, B_N,omega[i])
    #print(str(w_e) + "qwreetry")
    a = fn.getSectionAddedMass(x_curve,data, ratioB)
    b = fn.getSectionDamping(x_curve,w_e,data, ratioB)
    A = scipy.integrate.trapz(a,x)
    B = scipy.integrate.trapz(b,x)
    omega_n = math.sqrt(C33/A)
    k_ = (B/(2*A))/omega_n
    tuningFactor.append(w_e/omega_n)
    A_.append(A)
    B_.append(B)
    k.append(k_)


tune = np.array(tuningFactor)
k = np.array(k)
#print(k)
A = np.array(A_)
B = np.array(B_)
F = fn.getExcForce(omega,1,x,data)
F_e = np.array(F, dtype = 'float')
#print(F_e)
#print(A,B)

zeta_ = []
eps_ = []
for i in range(len(omega)):
    zeta, eps = fn.getResponse(M,A[i],B[i],C33,F_e[i], omega[i])
    zeta_.append(zeta)
    eps_.append(eps)

zeta = np.array(zeta_)
zeta_st = F_e/C33
zeta_final = np.zeros(len(zeta))
eps = np.array(eps_)
for i in range(len(zeta)):
    zeta_final[i] = abs(zeta[i]/zeta_st[i])
fn.plotSections(data,x)


fig = plt.figure(figsize=(8,4.5))
fig.suptitle("Heave RAO at Fn = 0.2 and μ = 180°")
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
ax1.plot(tune,zeta_final, color = 'black', label = 'RAO')
#ax1.set_xticks()
ax1.set_yticks(np.arange(0, 11, step=1))
#plt.title("Heave RAO at Fn = 0.2 and μ = 180°")
#plt.axhline(0, color = 'black',linewidth=0.8)
ax1.set_ylabel("RAO ζ/$ζ_{st}$")
ax1.set_xlabel("$ω_e$/$ω_n$")
ax2.set_ylabel("Phase Difference (°)")
ax1.grid(alpha = 0.3)
ax2.plot(tune, eps,  linestyle = '--', color = 'lightblue', label = 'Phase Difference')
ax1.legend(loc = 'upper left')
ax2.legend(loc = 'upper right')
ax2.set_ylim(-90,90)
ax1.set_ylim(0,10)
#plt.legend()
plt.show()
plt.plot(lamb, zeta_final)
plt.show()

