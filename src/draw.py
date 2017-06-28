# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
from numpy import where
from pylab import scatter, show, legend, xlabel, ylabel

# 获得方便绘图的数据
def normalize_Matrix():
    enemy = np.loadtxt('Enemy.csv', delimiter=",")
    teammate = np.loadtxt('Teammate_Rate.csv', delimiter=",")
    erate = [[0.0 for row in range(113)] for col in range(113)]
    frate = [[0.0 for row in range(113)] for col in range(113)]

    for i in range(113):
        for j in range(113):
            if enemy[i][j] != 0 and abs(enemy[i][j]) < 1:
                # erate = math.log(enemy[i][j] / basic[i][3]) * 100.0
                erate[i][j] = enemy[i][j] * 100
                # spamwriter.writerow([i, j, erate])
            if teammate[i][j] != 0 and abs(teammate[i][j]) < 1:
                # frate = math.log(teammate[i][j] / basic[i][3]) * 100.0
                frate[i][j] = teammate[i][j] * 100
                # spamwriter2.writerow([i, j, frate])
    return erate, frate

# 绘制三维散点图
def scatters(Data, fig, thresh):
    ax = fig.add_subplot(111, projection='3d')

    X = []
    Y = []
    Z = []
    C = []

    for i in range(113):
        for j in range(113):
            X.append(i)
            Y.append(j)
            Z.append(Data[i][j])

    for z in Z:
        if z > thresh:
            C.append('r')
        elif z < -1*thresh:
            C.append('b')
        else:
            C.append('y')

    ax.scatter(X, Y, Z, c=C, alpha=0.4, s=10)
    ax.set_xlabel('HeroA')
    ax.set_ylabel('HeroB')
    ax.set_zlabel('BetterRate')


if __name__ == '__main__':
    # 绘制协同系数和克制系数的三维散点图
    E, F = normalize_Matrix()
    fig1 = plt.figure()
    fig2 = plt.figure()
    scatter(F, fig1, 40)
    scatter(F, fig2, 30)
    fig1.show()
    fig2.show()
    input()

    # 读取测试集数据（如果绘制训练集则改为读训练集数据即可）
    test = np.loadtxt('Pre_Process_Test.csv', delimiter=',')
    X_test = test[:, 1:]
    Y_test = test[:, 0]

    # 绘制2属性模型的分类散点图
    pos = where(Y_test == 1)
    neg = where(Y_test == 0)
    scatter(X_test[pos, 0], X_test[pos, 1], marker='o', c='b')
    scatter(X_test[neg, 0], X_test[neg, 1], marker='x', c='r')
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')
    legend(['Not Admitted', 'Admitted'])
    show()
