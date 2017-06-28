import json
import numpy as np
import csv

def Mode_Distribution(X, Mode):
    countM = [0 for row in range(23)]

    for x in X:
        countM[int(x[2])] += 1

    for m in range(23):
        per = float(countM[m] * 100)/X.shape[0]
        print('%s: %d, %d%s' % (Mode[m], countM[m], per, "%"))

def Mode2Team(X):
    teammate = np.loadtxt('Teammate_Rate.csv', delimiter=",")
    Teamwork1 = []
    Teamwork2 = []
    for x in X:
        team1 = []
        team2 = []
        for id in range(0, 113):
            if x[id + 4] == 1:
                team1.append(id)
            elif x[id + 4] == -1:
                team2.append(id)
        teamwork = []
        for id in range(5):
            for tid in range(id+1, 5):
                if x[0] == 1:
                    teamwork.append(teammate[team1[id]][team1[tid]])
                else:
                    teamwork.append(teammate[team2[id]][team2[tid]])
        if x[2] == 2:
            Teamwork1.append(np.average(teamwork))
        elif x[2] == 8:
            Teamwork2.append(np.average(teamwork))
    print "Captain Mode: %f"%(np.average(Teamwork1))
    print "Reversed Captain Mode: %f"%(np.average(Teamwork2))


if __name__ == "__main__":
    dataset = np.loadtxt('dota2Train.csv', delimiter=",")
    X = dataset[:, 0:117]
    y = dataset[:, 0]


    Mode = ["Unknown", "All Pick", "Captain's Mode", "Random Draft", "Single Draft", "All Random", "Intro", "Diretide", "Reverse Captain's Mode", "The Greeviling", "Tutorial", "Mid Only", "Least Played", "New Player Pool", "Compendium Matchmaking", "Custom", "Captains Draft", "Balanced Draft", "Ability Draft", "Event", "All Random Death Match", "Solo Mid 1 vs 1", "Ranked All Pick"]

    # Mode_Distribution(X, Mode)
    Mode2Team(X)