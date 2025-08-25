def getLewisCurves(file_path):

    import numpy as np
    import matplotlib.pyplot as plt

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
        plt.plot(X_new, Y_new, label = "Bn/T = " + str(C[i]))
        data.append(np.row_stack((X_new,Y_new)))
        i+=1

    data = np.array(data)
    # plt.legend()
    # plt.show()
    return data