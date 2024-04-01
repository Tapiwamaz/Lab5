import numpy as np
import math
import matplotlib.pyplot as plt

# 1 a)
dist1X = np.random.normal(1,1,60)
dist1Y = np.random.normal(-1,1,60)

dist1Matrix = np.ones((len(dist1X),3))
for i in range(len(dist1X)):
    for j in range(1,3):
        if j ==1:
            dist1Matrix[i][j] = dist1X[i] 
        else:    
            dist1Matrix[i][j] = dist1Y[i]

dist1MatrixTraining = dist1Matrix[:20] 
dist1MatrixValidation = dist1Matrix[20:40]  
dist1MatrixTesting = dist1Matrix[40:]         

dist1XTraining = dist1X[:20]
dist1YTraining = dist1Y[:20]

dist1XTest = dist1X[20:]
dist1YTest = dist1Y[20:]

# 1 a)
dist2X = np.random.normal(-1,1,60)
dist2Y = np.random.normal(1,1,60)

dist2Matrix = np.ones((len(dist2X),3))
for i in range(len(dist2X)):
    for j in range(1,3):
        if j ==1:
            dist2Matrix[i][j] = dist2X[i] 
        else:    
            dist2Matrix[i][j] = dist2Y[i]

dist2MatrixTraining = dist2Matrix[:20]
dist2MatrixValidation = dist2Matrix[20:40]
dist2MatrixTesting = dist2Matrix[40:]

bothDistributionTraining = np.append(dist1MatrixTraining,dist2MatrixTraining).reshape(40,3)
bothDistributionValidation = np.append(dist1MatrixValidation,dist2MatrixValidation).reshape(40,3)
bothDistributionTesting = np.append(dist1MatrixTesting,dist2MatrixTesting).reshape(40,3)
# print(bothDistributionTraining)

dist2XTraining = dist2X[:20]
dist2YTraining = dist2Y[:20]

dist2XTest = dist2X[20:]
dist2YTest = dist2Y[20:]

theta = np.random.uniform(-0.5,0.5,3)


def createLineCoords(parameters, inputs ):
    coords = np.array([])
    for i in range(len(inputs)):
        y = (-parameters[1]*inputs[i] - parameters[0])/parameters[2] 
        coords = np.append(coords,y) 
    return coords    

# arr = np.array([-10,10])
# out = createLineCoords(theta,arr)


# graph = plt.scatter(dist1XTraining,dist1YTraining,color="black")
# graph = plt.scatter(dist2XTraining,dist2YTraining,color="red")
# graph = plt.plot(arr,out,marker="o")
# plt.xlim(-5,5)
# plt.ylim(-10,10)
# plt.show()


# function giving the probality of a coordinate being in a class 1
def logisticFunction(parameters,coord):
    xValue = np.dot(parameters,coord)
    result = 1 / (1 + pow(math.e,-xValue) )
    return result


def error(parameters, dataPoints, classNumber):
    total = 0
    for point in dataPoints:
        total += classNumber * math.log10((logisticFunction(parameters,point))) + (1 -classNumber)* math.log10(1-logisticFunction(parameters,point))  
    return -total

errDist1With0 = error(theta,dist1MatrixTraining,0)
errDist1With1 = error(theta,dist1MatrixTraining,1)
errDist2With0 = error(theta,dist2MatrixTraining,0)
errDist2With1 = error(theta,dist2MatrixTraining,1)
errorDist1 = min(errDist1With0 ,errDist1With1)
errorDist2 = min(errDist2With0 , errDist2With1)


def confusionMatrix(parameters,dataPoints,classNumber,confusionMatrix):
    for point in dataPoints:
        prediction = round(logisticFunction(parameters,point))
        if prediction == classNumber:
            confusionMatrix[classNumber][classNumber] += 1
        else:
            confusionMatrix[prediction][classNumber] +=1
    return confusionMatrix
confusion = np.zeros((2,2))
confusion = confusionMatrix(theta,dist1MatrixTraining,0,confusion) 
confusion = confusionMatrix(theta,dist2MatrixTraining,1,confusion) 

def sigmoid(predictions):
    return 1 / (1 + np.exp(-predictions))

def descent(parameters,dataPoints,alpha,y):
    linearPredictions = np.dot(dataPoints,parameters)
    predictions = sigmoid(linearPredictions)

    N  = len(dataPoints)
    X0,X1, X2 = dataPoints.T
    dw = 1/N * np.dot(np.array([X1,X2]), (predictions-y))
    db = 1/N* np.sum(predictions-y)
    
    for i in range(len(parameters)):
        if i !=0:
            parameters[i] = parameters[i] - alpha*dw[i-1]
        else: parameters[i] = parameters[i] - alpha*db
    
print("Training parameters....")
print("Theta before: ",theta)
print("Error ",errorDist1 + errorDist2)
print(confusion)

y = np.append(np.zeros(20),np.ones(20))
e = 0.05
count = 0
thetaOld = np.array(theta)
while np.linalg.norm(theta - thetaOld) < e  and count <3:
    thetaOld = theta 
    descent(theta,bothDistributionTraining,0.01,y)
    count+=1

print("Operations: ",count)

print("Theta after: ",theta)
errorDist1 = min(error(theta,dist1MatrixTraining,0),error(theta,dist1MatrixTraining,1))
errorDist2 = min(error(theta,dist2MatrixTraining,1),error(theta,dist2MatrixTraining,0))
print("Error ",errorDist1 + errorDist2)



confusion = np.zeros((2,2))
confusion = confusionMatrix(theta,dist1MatrixTraining,0,confusion)
confusion = confusionMatrix(theta,dist2MatrixTraining,1,confusion)
print(confusion)
print("Done....\n")

def doClassifying(parameters,dataPoints):
    confusion = np.zeros((2,2))
    confusion = confusionMatrix(parameters,dataPoints[:20],0,confusion)
    confusion = confusionMatrix(parameters,dataPoints[20:],1,confusion)
    print("Classified DataPoints:\n")
    errorDist1 = min(error(theta,dist1MatrixValidation,0),error(theta,dist1MatrixValidation,1))
    errorDist2 = min(error(theta,dist2MatrixValidation,1),error(theta,dist2MatrixValidation,0))
    print("Error ",errorDist1 + errorDist2)
    print(confusion)

print("Validations:\n")
doClassifying(theta,bothDistributionValidation)


# plt.close()
# graph = plt.scatter(dist1XTraining,dist1YTraining,color="black")
# graph = plt.scatter(dist2XTraining,dist2YTraining,color="red")
# out = createLineCoords(theta,arr)
# graph = plt.plot(arr,out,marker="o")
# plt.xlim(-5,5)
# plt.ylim(-10,10)
# plt.show()
