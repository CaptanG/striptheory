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
