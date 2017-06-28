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

if __name__ == "__main__":
    dataset = np.loadtxt('dota2Train.csv', delimiter=",")
    X = dataset[:, 0:117]
    y = dataset[:, 0]


    Mode = ["Unknown", "All Pick", "Captain's Mode", "Random Draft", "Single Draft", "All Random", "Intro", "Diretide", "Reverse Captain's Mode", "The Greeviling", "Tutorial", "Mid Only", "Least Played", "New Player Pool", "Compendium Matchmaking", "Custom", "Captains Draft", "Balanced Draft", "Ability Draft", "Event", "All Random Death Match", "Solo Mid 1 vs 1", "Ranked All Pick"]

    Mode_Distribution(X, Mode)
