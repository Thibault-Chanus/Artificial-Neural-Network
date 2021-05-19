import numpy as np
import typing as tp



class ANN(object):

    def __init__(self, struct):
        self.structure = struct
        self.ttstruct = []
        for i in range(0,len(self.structure) - 1):
            self.ttstruct.append(self.NL(self.structure[i],self.structure[i+1]))


    def forward(self, X):
        truc = X
        for a in self.ttstruct:
            truc = a.forward(truc)
        return truc

    def backward(self, X, y):
        self.o = self.forward(X)
        self.erreur = y - self.o
        self.erreur = np.append(self.erreur, [1])
        for i in range(0, len(self.ttstruct)):
            self.erreur = self.ttstruct[-i-1].backward1(self.erreur)
        truc = np.append(X, [1])
        for a in self.ttstruct:
            truc = a.backward2(truc)

    def train(self, X, y, nb ):

        for j in range(0,nb):
            for i in range( 0, len(X)):
                self.backward(X[i], y[i])

    class NL(object):
        def __init__(self, numavant, numapre):
            self.numavant = numavant
            self.numapre = numapre
            self.W1 = np.random.rand(self.numavant + 1, self.numapre)

        def forward(self, x):
            x_ = []
            try:
                for a in x:
                    x_.append(a)
            except TypeError:
                x_.append(x)
            x_.append(1)
            self.z = np.dot(x_,self.W1)
            self.z2 = self.fonction(self.z)
            self.z3 = np.array(self.z2, dtype=float)
            self.z2.append(1)
            return(self.z3)

        def fonction(self, x):
            A=[]
            for a in x:
                A.append(self.fonctionAct(a))
            return A

        def fonctionAct(self, x):
            return(1/(1+np.exp(-x)))

        def drv(self, x):
            x_ = []
            try:
                for a in x:
                    x_.append(self.fonctionDrv(a))
            except TypeError:
                x_.append(self.fonctionDrv(x))
            return np.array(x_, dtype=float)
        def fonctionDrv(self, x):
            return x * (1 -x)
        def lepop(self):
            _ = self.z2.pop()
        def backward1(self, erreur):
            self.erreur = np.array(erreur,dtype=float)
            self.delta = self.erreur * self.drv(self.z2)
            self.delto = self.delta[:-1]
            self.nextErreur = self.delto.dot(self.W1.T)
            return(self.nextErreur)

        def backward2(self, avant):
            self.before = np.array(avant, dtype=float)
            a = self.before.reshape(-1,1)
            b = self.delto.reshape(1,-1)
            self.W1 += a.dot(b)
            return(self.z2)


