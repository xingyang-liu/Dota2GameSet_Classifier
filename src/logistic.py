# coding=utf-8
import math
import numpy as np
from numpy import where
from pylab import scatter, show, legend, xlabel, ylabel

# sigmoid函数：sigmoid(x)=\frac{1}{1+ e^{-x}}
def Sigmoid(z):
    G_of_Z = float(1.0 / float((1.0 + math.exp(-1.0*z))))
    return G_of_Z

# h函数是所有属性的线性组合，并通过sigmoid函数将值域压缩到[0,1]
def Hypothesis(theta, x):
    z = 0
    for i in xrange(len(theta)):
        z += x[i]*theta[i]
    return Sigmoid(z)

# cost函数用于衡量h函数预测的结果是否合理
# J模型是cost函数对于所有点的平均值
def Cost_Function(X,Y,theta,m):
    sumOfErrors = 0
    for i in xrange(m):
        xi = X[i]
        hi = Hypothesis(theta,xi)
        if Y[i] == 1:
            error = Y[i] * math.log(hi)
        elif Y[i] == 0:
            error = (1-Y[i]) * math.log(1-hi)
        sumOfErrors += error
    const = -1/m
    J = const * sumOfErrors
    # print 'cost is ', J
    return J

# 对cost函数求偏导，为梯度下降法提供正确的方向
# a为学习率，决定了学习的步长
def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
    sumErrors = 0
    for i in xrange(m):
        xi = X[i]
        xij = xi[j]
        hi = Hypothesis(theta,X[i])
        error = (hi - Y[i])*xij
        sumErrors += error
    m = len(Y)
    constant = float(alpha)/float(m)
    J = constant * sumErrors
    return J

# 根据偏导的结果更新\theta
def Gradient_Descent(X,Y,theta,m,alpha):
    new_theta = []
    for j in xrange(len(theta)):
        CFDerivative = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
        new_theta_value = theta[j] - CFDerivative
        new_theta.append(new_theta_value)
    return new_theta

# 逻辑回归LR
def Logistic_Regression(X,Y,alpha,theta,num_iters):
    m = len(Y)
    for x in xrange(num_iters):
        new_theta = Gradient_Descent(X,Y,theta,m,alpha)
        theta = new_theta
        # 每迭代100次输出cost函数结果
        if x % 100 == 0:
            cost = Cost_Function(X,Y,theta,m)
            print cost
            # 每迭代1000次进行一次模型评估
            if x % 1000 == 0:
                LR_Test(X_test, Y_test, theta)

    return theta

# 用来对分类结果进行评估
def LR_Test(X_test, Y_test, theta, thresh=0.5):
    a = 0
    b = 0
    c = 0
    d = 0
    length = len(X_test)
    for i in xrange(length):
        hi = Hypothesis(X_test[i],theta)
        if hi > thresh:
            prediction = 1
        else:
            prediction = 0
        answer = Y_test[i]
        if prediction == 1:
            if answer == 1:
                a += 1
            else:
                c += 1
        else:
            if answer == 1:
                b += 1
            else:
                d += 1
    # 计算评估的指标
    accuracy = float(a + d) / float(length)
    precision = float(a) / float(a + c)
    recall = float(a) / float(a + b)

    print 'thresh: ', thresh
    print 'Accuracy: ', accuracy
    print 'Precision: ', precision
    print 'Recall: ', recall

if __name__ == "__main__":
    print('reading training and testing data...')
    # 读取训练集数据
    train = np.loadtxt('Pre_Process_Train.csv', delimiter=',')
    X_train = train[:, 1:]
    Y_train = train[:, 0]
    # 读取测试集数据
    test = np.loadtxt('Pre_Process_Test.csv', delimiter=',')
    X_test = test[:, 1:]
    Y_test = test[:, 0]

    # 设置默认参数并开始逻辑回归
    # 2属性模型将下面数字改为2， 6属性模型将下面数字改为6
    initial_theta = [0 for col in range(2)]
    alpha = 1
    iterations = 5000
    theta = Logistic_Regression(X_train, Y_train, alpha, initial_theta, iterations)
    # 测试分类结果
    LR_Test(X_test, Y_test, theta)

