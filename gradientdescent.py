import numpy as np

import math
import random
import matplotlib.pyplot as plt
from sklearn import datasets as skdata

theta=[]

def checkZero(data):
    if data==0:
        data=.0001
    return data 
#check performance
def checkperformance(test,ActualResult,theta):
    result=sigmoid(theta,test)
    result=[float(i) for i in result]
    for i in range(len(result)):
        if result[i]>.5:
            result[i]=1
        elif result[i]<.5:
            result[i]=0
    
    truePos=0
    trueNeg=0
    falsePos=0
    falseNeg=0
    for i in range(len(result)):
        if result[i]==ActualResult[i] and ActualResult[i]==1:
            truePos+=1
        elif result[i]==ActualResult[i] and ActualResult[i]==0:
            trueNeg+=1 
        elif result[i]!=ActualResult[i] and ActualResult[i]==1:
            falsePos+=1
        elif result[i]!=ActualResult[i] and ActualResult[i]==0:
            falseNeg+=1 
    truePos=checkZero(truePos)
    trueNeg=checkZero(trueNeg)
    falsePos=checkZero(falsePos)
    falseNeg=checkZero(falseNeg)        
    Precision=truePos/(truePos+falsePos)
    Recall=truePos/(truePos+falseNeg)
    FMeasure=(2*Precision*Recall)/(Precision+Recall)
    Accuracy= (truePos+trueNeg)/(truePos+trueNeg+falseNeg+falsePos)
    print("Precision:",Precision)
    print("Recall:",Recall)
    print("F-measure:",FMeasure)
    print("Accuracy:",Accuracy)
#calculate sigmoid func
def sigmoid(theta,x):
    return (1/(1+np.exp(-np.dot(x,theta))))
    
#calc gradient
def log_gradient(theta, x, y):
    first_calc = sigmoid(theta, x) -y
    final_calc = first_calc.T.dot(x)
    return final_calc

              

def addbias(data):
    return np.hstack([data,np.ones([data.shape[0],1])])

def cost_func(theta,x,y):
    
    return np.mean(-y*np.log(sigmoid(theta,x))-(1-y)*np.log(1 - sigmoid(theta,x)))
    
#standardise data
def standardize(data):
    mean= np.mean(data)
    std = np.std(data)
    xStd = (data-mean)/std
    return xStd

def initalTheta():
    for i in range(57):
        theta.append(random.uniform(-1,1))
    return theta

def main():
           
    data = np.genfromtxt('./spambase.data',delimiter=',')

    theta=[]
    #randomise data
    np.random.seed(0)
    np.random.shuffle(data)



    #divide data to test , train n class
    train_pct_index = int((2/3) * len(data))
    train,test = data[:train_pct_index], data[train_pct_index:]
    Y = train[:, train.shape[1] - 1:]
    X = train[:, :train.shape[1] - 1]
    ActualResult = test[:, test.shape[1] - 1:]
    test = test[:, :test.shape[1] - 1]
    Y=np.squeeze(Y)
    X=standardize(X)
    test=standardize(test)
    xt=X.transpose()
    #X=addbias(X)
    
    theta=initalTheta()
    theta=np.array(theta)
    
    sig = sigmoid(theta,X)
    
    pr=1
    
    itr = 0
    lr = 0.01
    N = len(X)
    lossOfData0 = 1
    #calculate theta
    while(itr<1500 and pr>2**-23):       
        lossOfData = cost_func(theta,X,Y)
        if math.isnan(lossOfData):
            pr=1
        else:
            
            pr=np.abs(lossOfData0-lossOfData)
            lossOfData0=lossOfData
        theta = theta - (lr * log_gradient(theta, X, Y))
        itr += 1

    checkperformance(test,ActualResult,theta)
   
    
if __name__ == '__main__':
    main()
