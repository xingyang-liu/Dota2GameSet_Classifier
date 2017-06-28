import json
import numpy as np
import csv
import math

def Basic_Rate(hero, X):
    csvfile = open('basic_rate.csv', 'wb')
    spamwriter = csv.writer(csvfile,dialect='excel')
    total_game = 0;
    select_game = [0]*113
    win_game = [0]*113

    for x in X:
        total_game += 1
        for id in range(0,113):
            select_game[id] += abs(x[id+4])
            if x[0] == x[id+4]:
                win_game[id] += 1

    for id in range(0, 113):
        select_rate = float(select_game[id])/float(total_game)
        if select_rate != 0:
            win_rate = float(win_game[id]) / float(select_game[id])
        else:
            win_rate = -1
        # print select_rate
        spamwriter.writerow([select_game[id], win_game[id], select_rate, win_rate])

def Enemy(hero, X):
    csvfile = open('Enemy.csv', 'wb')
    spamwriter = csv.writer(csvfile, dialect='excel')
    total_game = 0;
    for id in range(0, 113):
        win_game = [0]*113
        encounter_game = [0]*113
        encounter_rate = [0.0]*113
        for x in X:
            if x[id+4] == 0:
                continue
            for c_id in range(0, 113):
                if x[id+4] == x[c_id+4]*(-1):
                    encounter_game[c_id] += 1;
                    if x[0] == x[id+4]:
                        win_game[c_id] += 1;

        for c_id in range(0, 113):
            if encounter_game[c_id] != 0:
                encounter_rate[c_id] = float(win_game[c_id])/float(encounter_game[c_id])
            else:
                encounter_rate[c_id] = -1

        spamwriter.writerow(encounter_rate)

def Teammate_Rate(hero, X):
    csvfile = open('Teammate_Rate.csv', 'wb')
    spamwriter = csv.writer(csvfile, dialect='excel')
    for id in range(0, 113):
        win_game = [0] * 113
        encounter_game = [0] * 113
        win_rate = [0.0] * 113
        for x in X:
            if x[id + 4] == 0:
                continue
            for c_id in range(0, 113):
                if x[id + 4] == x[c_id + 4] and id != c_id:
                    encounter_game[c_id] += 1;
                    if x[0] == x[id + 4]:
                        win_game[c_id] += 1;

        for c_id in range(0, 113):
            if encounter_game[c_id] != 0:
                win_rate[c_id] = float(win_game[c_id]) / float(encounter_game[c_id])
            else:
                win_rate[c_id] = -1

        spamwriter.writerow(win_rate)

def Teammate_Frequency(hero, X):
    csvfile = open('Teammate_Frequency.csv', 'wb')
    csvfile2 = open('Teammate_Times.csv')
    spamwriter = csv.writer(csvfile, dialect='excel')
    spamwriter2 = csv.writer(csvfile2, dialect='excel')
    for id in range(0, 113):
        total_game = 0;
        encounter_game = [0] * 113
        encounter_rate = [0] * 123
        for x in X:
            if x[id + 4] == 0:
                continue
            total_game += 1
            for c_id in range(0, 113):
                if x[id + 4] == x[c_id + 4] and id != c_id:
                    encounter_game[c_id] += 1;

        for c_id in range(0, 113):
            if total_game != 0:
                encounter_rate[c_id] = float(encounter_game[c_id]) / float(total_game)
            else:
                encounter_rate[c_id] = 'N/A'

        spamwriter.writerow(encounter_game)
        spamwriter2.writerow(encounter_rate)

def normalize_Matrix(X):
    enemy = np.loadtxt('Enemy.csv', delimiter=",")
    teammate = np.loadtxt('Teammate_Rate.csv', delimiter=",")
    basic = np.loadtxt('basic_rate.csv', delimiter=",")
    csvfile = open('enemy_normalize.csv', 'wb')
    spamwriter = csv.writer(csvfile, dialect='excel')
    csvfile2 = open('friend_normalize.csv', 'wb')
    spamwriter2 = csv.writer(csvfile2, dialect='excel')

    for i in range(113):
        for j in range(113):
            if enemy[i][j] <= 0:
                continue
            erate = math.log(enemy[i][j] / basic[i][3]) * 100.0
            spamwriter.writerow([i, j, erate])
            if teammate[i][j] <= 0:
                continue
            frate = math.log(teammate[i][j] / basic[i][3]) * 100.0
            spamwriter2.writerow([i, j, frate])


def Pre_Process(X):
    enemy = np.loadtxt('Enemy.csv', delimiter=",")
    teammate = np.loadtxt('Teammate_Rate.csv', delimiter=",")
    basic = np.loadtxt('basic_rate.csv', delimiter=",")
    # Data = [[0 for col in range(55)] for row in range(X.shape[0])]
    Data = [[0 for col in range(10)] for row in range(X.shape[0])]
    i = 0
    for x in X:
        team1 = []
        team2 = []
        sid = 30
        for id in range(0, 113):
            if x[id + 4] == 1:
                team1.append(id)
            elif x[id + 4] == -1:
                team2.append(id)
        for id in range(5):
            Data[i][id*6:id*6+6] = [basic[team1[id]][3], enemy[team1[id]][team2[0]], enemy[team1[id]][team2[1]], enemy[team1[id]][team2[2]], enemy[team1[id]][team2[3]], enemy[team1[id]][team2[4]]]
            Data[i][id+50] = basic[team2[id]][3]
            for tid in range(id+1, 5):
                Data[i][sid] = teammate[team1[id]][team1[tid]]
                Data[i][sid+10] = teammate[team2[id]][team2[tid]]
                sid += 1
        i += 1

    # csvfile = open('Pre_Process.csv', "wb")
    # spamwriter = csv.writer(csvfile, dialect='excel')
    # for d in Data:
    #     spamwriter.writerow(d)

    return Data


if __name__ == "__main__":
    fp = open('heroes.json', 'r')
    hero = json.load(fp)

    dataset = np.loadtxt('dota2Train.csv', delimiter=",")
    X_train = dataset[:, 0:117]
    y_train = dataset[:, 0]

    test = np.loadtxt('dota2Test.csv', delimiter=',')
    X_test = test[:, 0:117]
    y_test = dataset[:, 0]

    # normalize_Matrix(X)

    # Basic_Rate(hero, X_train)
    # Enemy(hero, X_train)
    # Teammate_Rate(hero, X_train)
    Data_train = Pre_Process(X_train)
    Data_test = Pre_Process(X_test)

    csvfile1 = open('Pre_Process_Train.csv', "wb")
    spamwriter1 = csv.writer(csvfile1, dialect='excel')
    for d in Data_train:
        spamwriter1.writerow(d)

    csvfile2 = open('Pre_Process_Test.csv', "wb")
    spamwriter2 = csv.writer(csvfile2, dialect='excel')
    for d in Data_test:
        spamwriter2.writerow(d)