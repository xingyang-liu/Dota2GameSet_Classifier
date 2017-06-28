# coding=utf-8
import json
import numpy as np
import csv

# 统计英雄的基础胜率、出场率等数据
def Basic_Rate(X):
    csvfile = open('basic_rate.csv', 'wb')
    spamwriter = csv.writer(csvfile,dialect='excel')

    total_game = 0 # 总游戏场数
    select_game = [0]*113 # 指定英雄出战数
    win_game = [0]*113 #指定英雄胜场数

    # 统计英雄的出场数和胜场数
    for x in X:
        total_game += 1
        for id in range(0,113):
            select_game[id] += abs(x[id+4])
            if x[0] == x[id+4]:
                win_game[id] += 1

    # 计算英雄的出场率和胜场率，并写入文件
    for id in range(0, 113):
        select_rate = float(select_game[id])/float(total_game)
        if select_rate != 0:
            win_rate = float(win_game[id]) / float(select_game[id])
        else:
            win_rate = 0

        spamwriter.writerow([select_game[id], win_game[id], select_rate, win_rate])

# 统计英雄之间的克制系数
def Enemy(X):
    csvfile = open('Enemy.csv', 'wb')
    spamwriter = csv.writer(csvfile, dialect='excel')

    encounter_rate = [[0.0 for col in range(113)] for row in range(113)] # 记录英雄间的相遇胜率
    basic = np.loadtxt('basic_rate.csv', delimiter=',')
    win_rate = basic[:, 3] # 英雄胜率

    # 计算每个英雄对其它英雄的战绩
    for id in range(0, 113):
        win_game = [0]*113 # 该英雄对其它英雄的胜场数
        encounter_game = [0]*113 # 该英雄对其它英雄的出战数

        # 计算每场游戏碰到的英雄以及是否胜利
        for x in X:
            if x[id+4] == 0:
                continue
            for c_id in range(0, 113):
                if x[id+4] == x[c_id+4]*(-1):
                    encounter_game[c_id] += 1
                    if x[0] == x[id+4]:
                        win_game[c_id] += 1

        # 计算该英雄对其它英雄的胜率
        for c_id in range(0, 113):
            if encounter_game[c_id] != 0:
                encounter_rate[id][c_id] = float(win_game[c_id])/float(encounter_game[c_id])
            else:
                encounter_rate[id][c_id] = -1

    # 计算英雄之间的克制系数
    for id in range(113):
        better_rate = [0.0 for col in range(113)]
        for c_id in range(113):
            if encounter_rate[id][c_id] == -1:
                better_rate[c_id] = 0
            else:
                theoretical_rate = (win_rate[id] - win_rate[id]*win_rate[c_id])/(win_rate[id]+win_rate[c_id] - 2*win_rate[id]*win_rate[c_id])
                better_rate[c_id] = (encounter_rate[id][c_id] - theoretical_rate) / theoretical_rate
        spamwriter.writerow(better_rate)

# 统计英雄之间的协同系数
def Teammate_Rate(X):
    csvfile = open('Teammate_Rate.csv', 'wb')
    spamwriter = csv.writer(csvfile, dialect='excel')

    basic = np.loadtxt('basic_rate.csv', delimiter=',')
    win_rate = basic[:, 3]
    encounter_rate = [[0.0 for col in range(113)] for row in range(113)] # 记录英雄间的相遇胜率

    # 计算每个英雄跟其它英雄搭档的战绩
    for id in range(0, 113):
        win_game = [0] * 113 # 记录该英雄搭档的胜场数
        encounter_game = [0] * 113 # 记录该英雄搭档的出战数

        # 记录每场游戏遇见的队友英雄及其战绩
        for x in X:
            if x[id + 4] == 0:
                continue
            for c_id in range(0, 113):
                if x[id + 4] == x[c_id + 4] and id != c_id:
                    encounter_game[c_id] += 1
                    if x[0] == x[id + 4]:
                        win_game[c_id] += 1

        # 计算该英雄搭档其它英雄的胜率
        for c_id in range(0, 113):
            if encounter_game[c_id] != 0:
                encounter_rate[id][c_id] = float(win_game[c_id]) / float(encounter_game[c_id])
            else:
                encounter_rate[id][c_id] = -1

    # 计算英雄之间的协同系数
    for id in range(113):
        better_rate = [0.0 for col in range(113)]
        for c_id in range(113):
            if encounter_rate[id][c_id] == -1:
                better_rate[c_id] = 0
            else:
                theoretical_rate = (win_rate[id]*win_rate[c_id])/(win_rate[id]*win_rate[c_id] + (1-win_rate[id])*(1-win_rate[c_id]))
                better_rate[c_id] = (encounter_rate[id][c_id] - theoretical_rate) / theoretical_rate
        spamwriter.writerow(better_rate)

# 对数据进行预处理以供分类器使用
def Pre_Process(X):
    # 获取基本的胜率信息
    enemy = np.loadtxt('Enemy.csv', delimiter=",")
    teammate = np.loadtxt('Teammate_Rate.csv', delimiter=",")
    basic = np.loadtxt('basic_rate.csv', delimiter=",")
    Data = []

    for x in X:
        team1 = [] # team 1 里的英雄名单
        team2 = [] # team -1 里的英雄名单
        # 获取出战的英雄
        for id in range(0, 113):
            if x[id + 4] == 1:
                team1.append(id)
            elif x[id + 4] == -1:
                team2.append(id)
        # 计算队伍的TWR
        Personal_strength1 = normalize(np.average([basic[team1[0]][3], basic[team1[1]][3], basic[team1[2]][3], basic[team1[3]][3], basic[team1[4]][3]]), 0.4, 0.5)
        Personal_strength2 = normalize(np.average([basic[team2[0]][3], basic[team2[1]][3], basic[team2[2]][3], basic[team2[3]][3], basic[team2[4]][3]]), 0.4, 0.5)
        # 计算队伍的TAR和TOR
        overcome1 = []
        overcome2 = []
        teamwork1 = []
        teamwork2 = []
        for id in range(5):
            overcome1.append(np.max([enemy[team1[id]][0], enemy[team1[id]][1], enemy[team1[id]][2], enemy[team1[id]][3], enemy[team1[id]][4]]))
            overcome2.append(np.max([enemy[team2[id]][0], enemy[team2[id]][1], enemy[team2[id]][2], enemy[team2[id]][3], enemy[team2[id]][4]]))
            for tid in range(id+1, 5):
                teamwork1.append(teammate[team1[id]][team1[tid]])
                teamwork2.append(teammate[team2[id]][team2[tid]])
        Overcome1 = np.average(overcome1)
        Overcome2 = np.average(overcome2)
        Teamwork1 = np.average(teamwork1)
        Teamwork2 = np.average(teamwork2)
        # 6属性模型则将下面语句uncomment
        # Data.append([Personal_strength1, Teamwork1, Overcome1, Personal_strength2, Teamwork2, Overcome2])

        # 2属性模型则将下面语句uncomment
        Personal_strength1 = Personal_strength1*(1+Overcome1)*(1+Teamwork1)
        Personal_strength2 = Personal_strength2 * (1 + Overcome2) * (1 + Teamwork2)
        Data.append([Personal_strength1, Personal_strength2])

    # 将结果写入文件，需要时可以uncomment
    # csvfile = open('Pre_Process.csv', "wb")
    # spamwriter = csv.writer(csvfile, dialect='excel')
    # for d in Data:
    #     spamwriter.writerow(d)

    return Data

# 将数据标准化，使数据能正好分布在[0,1]之间
def normalize(data, range, ave):
    return (data-ave) * 1 / range + 0.5


if __name__ == "__main__":
    # 读取训练集
    dataset = np.loadtxt('dota2Train.csv', delimiter=",")
    X_train = dataset[:, 0:117]
    y_train = dataset[:, 0]
    # 读取测试集
    test = np.loadtxt('dota2Test.csv', delimiter=',')
    X_test = test[:, 0:117]
    y_test = test[:, 0]

    # 分别计算WR、OR和AR
    Basic_Rate(X_train)
    Enemy(X_train)
    Teammate_Rate(X_train)

    # 分别对训练集的数据和测试集的数据进行预处理
    Data_train = Pre_Process(X_train)
    Data_test = Pre_Process(X_test)

    # 将-1的label换成0
    for i in range(X_train.shape[0]):
        if y_train[i] == -1.0:
            y_train[i] = 0
    for i in range(X_test.shape[0]):
        if y_test[i] == -1.0:
            y_test[i] = 0

    # 将预处理的结果写入文件中供分类器使用
    csvfile1 = open('Pre_Process_Train.csv', "wb")
    spamwriter1 = csv.writer(csvfile1, dialect='excel')
    for i in range(X_train.shape[0]):
        spamwriter1.writerow([y_train[i]]+Data_train[i])

    csvfile2 = open('Pre_Process_Test.csv', "wb")
    spamwriter2 = csv.writer(csvfile2, dialect='excel')
    for i in range(X_test.shape[0]):
        spamwriter2.writerow([y_test[i]] + Data_test[i])