import sectionFunction as fn
import numpy as np

file_path1 = "amplitudeRatio_0.7.txt"
file_path2 = "addedMass_0.7.txt"

beta_n = fn.calcBeta()
addedMass, c1 = fn.getLewisCurves(file_path1)
amplitudeRatio, c2 = fn.getLewisCurves(file_path2)
print(addedMass[1,1])

x, data = fn.getSections("wig_seakeeping.txt")
T = abs(data[0,:]).max() #Only works for Wigley hull
B_N = 2*abs(data[:,0]).max()
B = []
for i in range(len(data)):
    B.append((2*fn.getMaxB(data[i,0]))/T)
ratioB = np.array(B)

rho = 1025
C = []
w = 1
for i in range(len(ratioB)):
    if(ratioB[i] == 0):
        C.append(0)
    else:
        idx = fn.find_closest_index(c1,ratioB[i])
        i2, i1 = fn.find_closest(addedMass[idx,0], w)
        c_new = fn.interpolate(addedMass[idx,0][i1],addedMass[idx,1][i1],addedMass[idx,0][i2],addedMass[idx,1][i2], x = w)
        C.append(c_new)
C = np.array(C)
print(C)
