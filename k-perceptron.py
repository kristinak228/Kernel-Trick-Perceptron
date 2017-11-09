'''
Kristina Kolibab
CS 449 - HW3
Kernel Perceptron
'''

import numpy as np
import time as t

def readData( file1, file2 ):

    # read in vector points
    f = open(file1, "r")
    lines = f.read()
    lines = lines.split('\n')
    f.close()
    trainList = []
    for line in lines:
        curr_line = line.split(' ')
        vec = curr_line
        trainList.append(vec)
    trainList = np.array(trainList)
    trainList = trainList[:-1]

    # read in true labels
    f2 = open(file2, "r")
    lines = f2.read()
    lines = lines.split('\n')
    f2.close()
    trueAnswer = []
    for line in lines:
        trueAnswer.append(line)
    trueAnswer = np.array(trueAnswer)
    trueAnswer = trueAnswer[:-1]
    return trainList, trueAnswer

def threshold(x):
    if x >= 0.0:
        return 1.0
    else:
        return -1.0

def K(x1, x2): 
    b = 1 # kernel bias
    x1 = [float(i) for i in x1]
    x2 = [float(i) for i in x2]
    return ((np.dot(x1, x2)) + b)**2

def eval():
    trainList, label = readData("train.txt", "train-label.txt")
    label = np.array(label) # have to convert 
    
    # loop through every point, per point
    gram = np.zeros(shape=(len(trainList), len(trainList)), dtype=float)       
    for i in range(len(trainList)):
        for j in range(len(trainList)):
            gram[i,j] = K(trainList[i], trainList[j])

    # dual weights
    b = 0
    alpha = np.zeros(len(trainList))
    
    omega = np.zeros(shape=(len(trainList), len(trainList)), dtype=float)
    for i in range(len(trainList)):
        for j in range(len(trainList)):
            omega[i,j] = float(label[j]) * float(gram[i,j])
    
    mistake = True
    while mistake:
        err = 0
        mistake = False
        for i in range(len(trainList)):
            if float(label[i]) * (np.dot(alpha, omega[i]) + b) <= 0.0: #!= float(label[i]):
                mistake = True
                alpha[i] = alpha[i] + 1.0
                b = b + float(label[i])
                err += 1
                print err

# M A I N
def main():
    eval()

if __name__ == "__main__":
    main()
