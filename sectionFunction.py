def getSections(file_path):

    import  numpy as np

    fp = open(file_path,'r')

    a = fp.readline().rstrip('\n').split(',')

    countSec = int(a[0])

    data = []
    X1 = []

    for i in range(countSec):

        a1 = fp.readline().rstrip('\n').split(',')
        n = int(a1[0])
        x = float(a1[1])
        X1.append(float(a1[1]))

        Y = []
        Z = []

        for j in range(n):
            a2 = fp.readline().rstrip('\n').split(',')
            Y.append(float(a2[0]))
            Z.append(float(a2[1]))
        Y=np.array(Y)
        Z=np.array(Z)
        X=np.zeros(n)+x
        # ax.plot3D(X,Y,Z, color = '#656565')
        # ax.plot3D(X,-Y,Z, color = '#656565')

        data.append(np.row_stack((Y,Z)))

    X = np.array(X1)

    data = np.array(data)

    fp.close()
    return(X, data)

def plotSections(data,x):
    
    import numpy as np
    from mpl_toolkits import mplot3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    
    fig = plt.figure(1)
    ax = plt.axes(projection='3d')

    for i in range(len(x)):
        Y = data[i,0,:]
        Z = data[i,1,:]
        X = np.zeros_like(Y) + x[i]
        ax.plot3D(X,Y,Z, color = '#656565')
        ax.plot3D(X,-Y,Z, color = '#656565')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-80,80)
    ax.set_ylim(-12,12)
    ax.set_zlim(-8,5)

    plt.show()

    return

def getVolume(data,x):
    
    import numpy as np
    import scipy

    secArea1 = []

    for i in range(len(x)):
        Y = data[i,0,:]
        Z = data[i,1,:]
        secArea1.append(2*abs(scipy.integrate.trapz(Y,Z)))

    secArea = np.array(secArea1)

    volume = scipy.integrate.trapz(secArea,x)

    return volume

def getC33(data,x):
    
    import numpy as np
    import scipy

    density = 1.025
    g = 9.81
    newY = []
    newY.append(abs(data[:,0,0]))
    Y = np.array(newY)
    C33 = density*g*2*scipy.integrate.trapz(Y,x)

    return C33

def getC35(data,x):

    import numpy as np
    import scipy

    density = 1.025
    g = 9.81
    newY = []
    newY.append(abs(data[:,0,0]))
    Y = np.array(newY)
    x = x + 60
    C35 = density*g*2*scipy.integrate.trapz(Y*x,x)

    return C35

def getC55(data,x):

    import numpy as np
    import scipy

    density = 1.025
    g = 9.81
    newY = []
    newY.append(abs(data[:,0,0]))
    Y = np.array(newY)
    C55 = density*g*2*scipy.integrate.trapz(Y*x*x,x)

    return C55 

def getExcForce(omega, amplitude, x, data):

    import numpy as np
    import scipy
    import matplotlib.pyplot as plt
    import math

    B_N = 2*abs(data[:,0]).max()
    g = 9.81
    rho = 1.025
    newY = []
    newY.append(abs(data[:,0,0]))
    Y = np.array(newY)
    t = np.zeros(len(omega))
    F = []

    for i in range(len(omega)):
        w_e, x_curve = inputFunc(x, B_N,omega[i])
        t[i] = (2*3.14)/(omega[i])
        Awp = 2*scipy.integrate.trapz(Y,x)
        F_o = 2*rho*g*amplitude*Awp
        total = F_o*math.cos(w_e*t[i])
        #k = (omega**2)/g
        #eta = amplitude*np.cos(k*x)
        #secF = 2*Y*eta*rho*g
        #total = scipy.integrate.trapz(secF,x)
        F.append(total)
    #fig = plt.figure()
    #ax = plt.axes()
    #plt.plot(omega,F)
    #plt.legend()
    #plt.show()

    return F 

def interpolate(x1,y1,x2,y2,y=None,x=None):
        if (x2-x1)==0:
            return x1
        m=(y2-y1)/(x2-x1)
        if y is not None:
            x=((y-y1)/(m))+x1
            return x
        if x is not None:
            y=((x-x1)*m)+y1
            return y

def find_closest(arr,value):
            
        for i in range(len(arr)):
            if value<arr[i]:
                break
        if (i==0 ):
            idx1=i
            idx2=i+1
            return idx2, idx1
        idx1=i-1
        idx2=i
        return idx2,idx1

def getLewisCurves(file_path):

    import numpy as np
    import matplotlib.pyplot as plt

    #file_path = "amplitudeRatio_0.7.txt"
    fp = open(file_path,'r')
    a = fp.readline().rstrip("\n")
    curveCount = int(a[0])
    #print(curveCount)
    C = []
    X_new = np.linspace(0, 1.5, 100)

    data = []
    i = 0
    while(i<curveCount):
        a = fp.readline().rstrip("\n").split("\t")

        if a[0] == '':
            break

        C.append(a)
        c = fp.readline().rstrip("\n")
        X = []
        Y = []

        for j in range(int(c)):
            a1 = fp.readline().rstrip("\n").split("\t")
            X.append(a1[0])
            Y.append(a1[1])

        Y = np.array(Y, dtype = 'float')
        X = np.array(X, dtype = 'float')
        Y_new = []

        for j in range(len(X_new)):
            i2, i1 = find_closest(X, X_new[j])
            y = interpolate(X[i1], Y[i1], X[i2], Y[i2], x = X_new[j])
            Y_new.append(y)

        Y_new = np.array(Y_new, dtype = 'float')
        #print(X_new,Y_new)
        a = fp.readline()
        #plt.plot(X_new, Y_new, label = "Bn/T = " + str(C[i]))
        data.append(np.row_stack((X_new,Y_new)))
        i+=1

    C = np.array(C, dtype = 'float')
    data = np.array(data)
    #plt.legend()
    # plt.show()
    return data,C

def calcBeta():

    import sectionFunction as fn
    import numpy as np
    import scipy

    x, data = fn.getSections("wig_seakeeping.txt")

    maxB = []
    for i in range(len(data)):
        maxB.append(2*getMaxB(data[i,0]))

    maxB = np.array(maxB)
    secArea1 =[]
    for i in range(len(data)):
            Y = data[i,0,:]
            Z = data[i,1,:]
            secArea1.append(2*abs(scipy.integrate.trapz(Y,Z)))

    secArea = np.array(secArea1)
    T = abs(data[0,:]).max() #Only works for Wigley hull
    #print(T)
    beta_n = []
    for i in range(len(data)):
        if(maxB[i] == 0):
            beta_n.append(0)
        else:
            beta_n.append(secArea[i]/(maxB[i]*T))

    beta_n = np.array(beta_n)

    return beta_n

def getMaxB(array):
        import numpy as np
        abs_max_value = np.abs(array).max()
        return abs_max_value

def find_closest_index(array, target):
    closest_index = None
    min_difference = float('inf')  # Initialize with infinity
    
    for i, num in enumerate(array):
        difference = abs(num - target)
        if difference < min_difference:
            min_difference = difference
            closest_index = i
    
    return closest_index

def getSectionAddedMass(x_,data,ratioB):


    import numpy as np

    file_path1 = "amplitudeRatio_0.7.txt"
    file_path2 = "addedMass_0.7.txt"

    addedMass, c1 = getLewisCurves(file_path1)
    amplitudeRatio, c2 = getLewisCurves(file_path2)

    #x, data = getSections("wig_seakeeping.txt")
    T = abs(data[0,:]).max() #Only works for Wigley hull
    B_N = 2*abs(data[:,0]).max()
   

    C = []
    for i in range(len(ratioB)):
        if(ratioB[i] == 0):
            C.append(0)
        else:
            idx = find_closest_index(c1,ratioB[i])
            i2, i1 = find_closest(addedMass[idx,0], x_)
            c_new = interpolate(addedMass[idx,0][i1],addedMass[idx,1][i1],addedMass[idx,0][i2],addedMass[idx,1][i2], x = x_)
            C.append(c_new)
    C = np.array(C)
    rho = 1.025
    a = (C*rho*B_N*B_N*3.14)/8

    return a

def inputFunc(x,B_N,w):
    import math
    #w = float(input("Please enter an omega: "))
    #Fn = float(input("Please enter Froude Number: "))
    Fn = 0.2
    L = 2*x.max()
    g = 9.81
    v = Fn*math.sqrt(g*L)
    w_e = w + ((w*w*v)/g)
    x_curve = (w_e*w_e*B_N)/(2*g)

    return w_e, x_curve

def getSectionDamping(x_,w_e,data,ratioB):

    import numpy as np

    file_path1 = "amplitudeRatio_0.7.txt"
    file_path2 = "addedMass_0.7.txt"

    addedMass, c1 = getLewisCurves(file_path1)
    amplitudeRatio, c2 = getLewisCurves(file_path2)

    #x, data = getSections("wig_seakeeping.txt")
    T = abs(data[0,:]).max() #Only works for Wigley hull
    B_N = 2*abs(data[:,0]).max()
    

    C = []
    for i in range(len(ratioB)):
        if(ratioB[i] == 0):
            C.append(0)
        else:
            idx = find_closest_index(c2,ratioB[i])
            i2, i1 = find_closest(amplitudeRatio[idx,0], x_)
            c_new = interpolate(amplitudeRatio[idx,0][i1],amplitudeRatio[idx,1][i1],amplitudeRatio[idx,0][i2],amplitudeRatio[idx,1][i2], x = x_)
            #print(c_new)
            C.append(c_new)
    C = np.array(C)
    rho = 1.025
    b = (rho*9.81*9.81*C*C)/(w_e**3)

    return b

def getResponse(M,A,B,C,F_e, omega):
    import numpy as np
    import math

    D = ((C - (M+A)*(omega**2))**2) + ((omega*B)**2)
    p = C - ((M+A)*(omega**2))
    q = -1*omega*B
    P = (F_e*p)/D
    Q = (F_e*q)/D
    zeta_amp = math.sqrt((P**2) + (Q**2))
    epsilon = math.atan(Q/P)*(180/math.pi)
    return zeta_amp, epsilon

