import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
import numpy as np

def scatter(Data, fig):
    ax = fig.add_subplot(111, projection='3d')

    X = Data[:,0]
    Y = Data[:,1]
    Z = Data[:,2]
    C = []

    for z in Z:
        if z > 50:
            C.append('r')
        elif z < -50:
            C.append('b')
        else:
            C.append('y')

    ax.scatter(X, Y, Z, c=C, alpha=0.4, s=10)
    ax.set_xlabel('HeroA')
    ax.set_ylabel('HeroB')
    ax.set_zlabel('BetterRate')


if __name__ == '__main__':
    F = np.loadtxt('friend_normalize.csv', delimiter=',')
    E = np.loadtxt('enemy_normalize.csv', delimiter=',')
    fig1 = plt.figure()
    fig2 = plt.figure()
    scatter(F, fig1)
    scatter(E, fig2)
    fig1.show()
    fig2.show()
    input()
